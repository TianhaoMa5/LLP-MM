import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from plench.core import algorithms, hparams_registry
from plench.core.networks import Featurizer
from plench.data.twitter_ethnicity_2017 import (
    CLASS_NAMES,
    FEATURE_DIM,
    TwitterCountyBagDataset,
    TwitterEvaluationDataset,
    bag_diagnostics,
    build_county_bags,
    build_twitter_ethnicity_loaders,
    load_twitter_ethnicity_bundle,
)
from plench.scripts.build_twitter_census_2012 import build, build_race3_proportions
from plench.scripts.construct_twitter_ethnicity_2017 import (
    _full_size_profile_url,
    _validated_image,
    extract_candidates,
    parse_historical_tweet,
)


def _write_processed(root: Path, *, misalign=False, overlap=False) -> Path:
    processed = root / "processed"
    processed.mkdir(parents=True)
    train_counties = ["01001"] * 7 + ["06037"] * 3
    train_ids = [f"train-{index}" for index in range(len(train_counties))]
    eval_ids = ["eval-0", "eval-1", "eval-2"]
    twitter_train = [f"twitter-train-{index}" for index in range(len(train_ids))]
    twitter_eval = [twitter_train[0] if overlap else "twitter-eval-0", "twitter-eval-1", "twitter-eval-2"]
    manifest = pd.DataFrame(
        {
            "instance_id": train_ids + eval_ids,
            "twitter_user_id": twitter_train + twitter_eval,
            "county_fips": train_counties + ["", "", ""],
            "image_path": [f"images/{value}.jpg" for value in train_ids + eval_ids],
            "text_path": [""] * 13,
            "split": ["train"] * 10 + ["evaluation"] * 3,
            "label": [np.nan] * 10 + [0, 1, 2],
        }
    )
    manifest.to_csv(processed / "manifest.csv", index=False)
    pd.DataFrame(
        {
            "county_fips": ["01001", "06037"],
            "p_white": [0.7, 0.4],
            "p_black": [0.2, 0.1],
            "p_hispanic": [0.1, 0.5],
        }
    ).to_csv(processed / "county_proportions.csv", index=False)
    (processed / "metadata.json").write_text(
        json.dumps({"normalize_race3_proportions": True}), encoding="utf-8"
    )
    rng = np.random.default_rng(4)
    np.save(processed / "xception_features.npy", rng.normal(size=(13, FEATURE_DIM)).astype(np.float32))
    ids = np.asarray(train_ids + eval_ids)
    if misalign:
        ids[[0, 1]] = ids[[1, 0]]
    np.save(processed / "instance_ids.npy", ids)
    return root


def _census_source_fixture() -> pd.DataFrame:
    rows = []
    for state, county, year, age_group, population, counts in [
        (1, 1, 5, 0, 1000, (600, 200, 150)),
        (6, 37, 5, 0, 2000, (800, 300, 700)),
        (1, 1, 4, 0, 900, (550, 180, 120)),
        (1, 1, 5, 1, 100, (60, 20, 15)),
    ]:
        white, black, hispanic = counts
        rows.append(
            {
                "SUMLEV": 50,
                "STATE": state,
                "COUNTY": county,
                "YEAR": year,
                "AGEGRP": age_group,
                "TOT_POP": population,
                "NHWAC_MALE": white // 2,
                "NHWAC_FEMALE": white - white // 2,
                "NHBAC_MALE": black // 2,
                "NHBAC_FEMALE": black - black // 2,
                "H_MALE": hispanic // 2,
                "H_FEMALE": hispanic - hispanic // 2,
            }
        )
    return pd.DataFrame(rows)


def test_historical_archive_parser_requires_exact_coordinates_and_profile_image():
    tweet = {
        "id_str": "tweet-1",
        "created_at": "historical-date",
        "coordinates": {"type": "Point", "coordinates": [-86.6, 32.5]},
        "user": {
            "id_str": "user-1",
            "profile_image_url_https": "https://pbs.twimg.com/profile_images/1/avatar_normal.jpg",
        },
        "text": "archived text",
    }
    parsed = parse_historical_tweet(tweet)
    assert parsed is not None
    assert parsed["twitter_user_id"] == "user-1"
    assert parsed["longitude"] == pytest.approx(-86.6)
    assert _full_size_profile_url(parsed["profile_image_url"]).endswith("avatar.jpg")

    place_only = {**tweet, "coordinates": None, "place": {"id": "not-used"}}
    assert parse_historical_tweet(place_only) is None
    default_image = json.loads(json.dumps(tweet))
    default_image["user"]["profile_image_url_https"] = (
        "https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png"
    )
    assert parse_historical_tweet(default_image) is None


def test_archive_candidates_map_users_to_natural_county_without_labels(tmp_path):
    geopandas = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    county_file = tmp_path / "counties.geojson"
    counties = geopandas.GeoDataFrame(
        {"GEOID": ["01001", "06037"]},
        geometry=[
            shapely_geometry.box(-87.0, 32.0, -86.0, 33.0),
            shapely_geometry.box(-119.0, 33.0, -117.0, 35.0),
        ],
        crs="EPSG:4326",
    )
    counties.to_file(county_file, driver="GeoJSON")

    def record(tweet_id, user_id, longitude, latitude):
        return {
            "id_str": tweet_id,
            "created_at": "historical-date",
            "coordinates": {"type": "Point", "coordinates": [longitude, latitude]},
            "user": {
                "id_str": user_id,
                "profile_image_url_https": (
                    f"https://pbs.twimg.com/profile_images/{user_id}/avatar_normal.jpg"
                ),
            },
            "text": f"text-{tweet_id}",
        }

    archive = tmp_path / "historical.jsonl.bz2"
    import bz2

    with bz2.open(archive, "wt", encoding="utf-8") as stream:
        for payload in [
            record("1", "same-user", -86.6, 32.5),
            record("2", "same-user", -86.7, 32.6),
            record("3", "other-user", -118.2, 34.0),
            record("4", "outside-us", 0.0, 0.0),
        ]:
            stream.write(json.dumps(payload) + "\n")

    output_path = extract_candidates(
        [archive], county_file, tmp_path / "dataset", write_observed_text=True
    )
    output = pd.read_csv(output_path, dtype={"county_fips": str, "twitter_user_id": str})
    assert output["twitter_user_id"].tolist() == ["other-user", "same-user"]
    assert output.set_index("twitter_user_id")["county_fips"].to_dict() == {
        "other-user": "06037",
        "same-user": "01001",
    }
    assert "label" not in output.columns
    assert output["instance_id"].is_unique
    provenance = json.loads(
        (tmp_path / "dataset/raw/archive_candidates.provenance.json").read_text()
    )
    assert provenance["current_x_api_queried"] is False
    assert provenance["construction"] == "historical_reconstruction_not_author_release"


def test_downloaded_profile_image_is_decoded_and_canonicalized():
    from io import BytesIO

    source = BytesIO()
    Image.new("RGBA", (16, 12), (10, 20, 30, 255)).save(source, format="PNG")
    encoded, suffix = _validated_image(source.getvalue())
    assert suffix == ".jpg"
    with Image.open(BytesIO(encoded)) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (16, 12)


def test_official_census_source_is_converted_without_silent_race3_normalization(tmp_path):
    source = _census_source_fixture()
    proportions = build_race3_proportions(source)
    assert proportions["county_fips"].tolist() == ["01001", "06037"]
    first = proportions.iloc[0]
    assert first["p_white"] == pytest.approx(0.6)
    assert first["p_black"] == pytest.approx(0.2)
    assert first["p_hispanic"] == pytest.approx(0.15)
    assert first[["p_white", "p_black", "p_hispanic"]].sum() == pytest.approx(0.95)

    source_path = tmp_path / "cc-est2012-alldata.csv"
    source.to_csv(source_path, index=False)
    raw = tmp_path / "dataset" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame({"county_fips": ["06037"]}).to_csv(raw / "users.csv", index=False)
    output_path = build(tmp_path / "dataset", input_csv=source_path)
    output = pd.read_csv(output_path, dtype={"county_fips": str})
    assert output["county_fips"].tolist() == ["06037"]
    provenance = json.loads(
        (raw / "census_county_proportions.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["race3_renormalized"] is False
    assert provenance["twitter_records_created"] is False


def test_manifest_feature_alignment_and_feature_dimension(tmp_path):
    bundle = load_twitter_ethnicity_bundle(_write_processed(tmp_path))
    assert bundle.features.shape == (13, FEATURE_DIM)
    assert bundle.num_classes == len(CLASS_NAMES) == 3
    with pytest.raises(ValueError, match="not exactly aligned"):
        load_twitter_ethnicity_bundle(_write_processed(tmp_path / "bad", misalign=True))


def test_county_chunking_is_deterministic_and_never_mixes_counties(tmp_path):
    bundle = load_twitter_ethnicity_bundle(_write_processed(tmp_path))
    first = build_county_bags(bundle, bundle.train_indices, max_bag_size=4, seed=11)
    second = build_county_bags(bundle, bundle.train_indices, max_bag_size=4, seed=11)
    assert [bag.bag_id for bag in first] == [bag.bag_id for bag in second]
    assert [bag.indices.tolist() for bag in first] == [bag.indices.tolist() for bag in second]
    assert max(len(bag.indices) for bag in first) <= 4
    assert sorted(len(bag.indices) for bag in first) == [3, 3, 4]
    for bag in first:
        counties = set(bundle.manifest.iloc[bag.indices]["county_fips"])
        assert counties == {bag.county_fips}
        assert bag.proportions.shape == (3,)
        assert np.isfinite(bag.proportions).all()
        assert (bag.proportions >= 0).all()
        assert bag.proportions.sum() == pytest.approx(1.0)
    report = bag_diagnostics(first)
    assert report["instances"] == 10
    assert report["bags"] == 3


def test_training_loader_hides_labels_and_evaluation_is_separate(tmp_path):
    root = _write_processed(tmp_path)
    train_loader, _, bundle, input_shape = build_twitter_ethnicity_loaders(
        str(root), max_bag_size=4, batch_size=8, seed=3, num_workers=0
    )
    views, proportions, indices, bag_ids, hidden_labels = next(iter(train_loader))
    assert input_shape == FEATURE_DIM
    assert views[0].shape[0] == 1
    assert views[0].shape[2] == FEATURE_DIM
    assert len(proportions) == 3
    assert torch.all(hidden_labels == -1)
    assert not hasattr(train_loader.dataset, "labels")
    evaluation = TwitterEvaluationDataset(bundle)
    assert [evaluation[index][1] for index in range(len(evaluation))] == [0, 1, 2]
    assert set(bundle.train_indices).isdisjoint(bundle.evaluation_indices)


def test_twitter_feature_mlp_forward_and_configs(tmp_path):
    hparams = hparams_registry.default_hparams("PM", "TwitterEthnicity2017")
    model = Featurizer(FEATURE_DIM, hparams)
    features = model(torch.zeros(2, FEATURE_DIM))
    assert features.shape == (2, 500)
    linear = Featurizer(FEATURE_DIM, {**hparams, "model": "Linear"})
    assert linear(torch.zeros(2, FEATURE_DIM)).shape == (2, FEATURE_DIM)

    config_dir = Path(__file__).parents[1] / "configs" / "twitter_ethnicity_2017"
    configs = sorted(config_dir.glob("*.json"))
    assert len(configs) == 4
    observed = set()
    for path in configs:
        config = json.loads(path.read_text())
        observed.add(config["bagsize"])
        assert config["dataset"] == "TwitterEthnicity2017"
        assert config["n_classes"] == 3
        assert config["batchsize"] == 1
    assert observed == {32, 64, 128, 256}


def test_pm_accepts_a_smaller_final_county_chunk(tmp_path):
    root = _write_processed(tmp_path)
    train_loader, _, _, _ = build_twitter_ethnicity_loaders(
        str(root), max_bag_size=4, batch_size=1, seed=1, num_workers=0
    )
    hparams = hparams_registry.default_hparams("PM", "TwitterEthnicity2017")
    hparams.update({"lr": 1e-3, "weight_decay": 0.0})
    model = algorithms.PM(
        epochs=4,
        input_shape=FEATURE_DIM,
        train_givenY=train_loader.dataset.label_prob,
        hparams=hparams,
        bagsize=4,
    )
    views, proportions, *_ = next(iter(train_loader))
    current_size = int(views[0].shape[1])
    model.bagsize = current_size
    features = views[0][0]
    target = torch.stack(proportions, dim=1).to(torch.float32)
    output = model.update((features, target))
    assert np.isfinite(output["loss"])


def test_training_evaluation_user_overlap_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        load_twitter_ethnicity_bundle(_write_processed(tmp_path, overlap=True))


def test_train_entrypoint_smoke_with_variable_county_chunks(tmp_path):
    root = _write_processed(tmp_path / "data")
    output = tmp_path / "output"
    config = {
        "dataset": "TwitterEthnicity2017",
        "data_dir": str(root),
        "algorithm": "PM",
        "bagsize": 4,
        "batchsize": 1,
        "n_classes": 3,
        "bag_build": "random",
        "holdout_fraction": 0.0,
        "hparams": json.dumps(
            {"model": "MLP", "lr": 0.001, "weight_decay": 0.0}
        ),
        "epochs": 1,
        "checkpoint_freq": 100,
        "seed": 0,
        "num_workers": 0,
        "skip_model_save": True,
        "output_dir": str(output),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "plench.train", "--config", str(config_path)],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    final = records[-1]
    assert final["test_acc"] == pytest.approx(final["test_accuracy"])
    assert "test_macro_f1" in final
    assert "test_weighted_f1" in final
    assert np.asarray(final["test_confusion_matrix"]).shape == (3, 3)
