# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the nanoWeather feature extractor and time features."""

import numpy as np

from feature_extraction_nanoweather import NUM_TIME_FEATURES, NanoWeatherFeatureExtractor, get_time_features


LATITUDE = -87.1875 + 5.625 * np.arange(32)
LONGITUDE = 5.625 * np.arange(64)


def nearest(values, target):
    return int(np.argmin(np.abs(values - target)))


def test_normalize_round_trip_and_save_load(tmp_path):
    feature_extractor = NanoWeatherFeatureExtractor(variables=["a", "b"], mean=[1.0, -2.0], std=[2.0, 0.5])
    states = np.random.default_rng(0).normal(size=(3, 2, 4, 8)).astype(np.float32)
    np.testing.assert_allclose(feature_extractor.denormalize(feature_extractor.normalize(states)), states, atol=1e-5)
    feature_extractor.save_pretrained(tmp_path)
    assert NanoWeatherFeatureExtractor.from_pretrained(tmp_path).to_dict() == feature_extractor.to_dict()


def test_time_features_shape():
    times = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]")
    features = get_time_features(times, LATITUDE, LONGITUDE)
    assert features.shape == (2, NUM_TIME_FEATURES, 32, 64)
    assert features.dtype == np.float32


def test_sunlight_at_known_times_and_places():
    times = np.array(["2020-06-21T12", "2020-06-21T00", "2020-12-21T12"], dtype="datetime64[ns]")
    sunlight = get_time_features(times, LATITUDE, LONGITUDE)[:, 0]
    tropic_north, tropic_south = nearest(LATITUDE, 23.4), nearest(LATITUDE, -23.4)
    greenwich, dateline = nearest(LONGITUDE, 0), nearest(LONGITUDE, 180)
    assert sunlight[0, tropic_north, greenwich] > 0.99  # June solstice noon at Greenwich: sun overhead
    assert sunlight[1, tropic_north, greenwich] == 0  # midnight at Greenwich: night
    assert sunlight[1, tropic_north, dateline] > 0.99  # but noon on the other side of the globe
    assert sunlight[2, tropic_south, greenwich] > 0.99  # December solstice: overhead in the southern tropic
    assert sunlight[0, -1].min() > 0  # polar day in the north in June
    assert sunlight[0, 0].max() == 0  # polar night in the south in June


def test_sunlight_end_of_step_is_six_hours_later():
    times = np.array(["2020-03-20T06"], dtype="datetime64[ns]")
    features = get_time_features(times, LATITUDE, LONGITUDE)
    later = get_time_features(times + np.timedelta64(6, "h"), LATITUDE, LONGITUDE)
    np.testing.assert_allclose(features[:, 1], later[:, 0])