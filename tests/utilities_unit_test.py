# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation




import attr
import numpy as np
import pytest

from floris.utilities import (
    cosd,
    sind,
    tand,
    wrap_180,
    wrap_360,
    is_default,
    float_attrib,
    model_attrib,
    iter_validator,
    attrs_array_converter,
)

def test_cosd():
    assert pytest.approx(cosd(0.0)) == 1.0
    assert pytest.approx(cosd(90.0)) == 0.0
    assert pytest.approx(cosd(180.0)) == -1.0
    assert pytest.approx(cosd(270.0)) == 0.0


def test_sind():
    assert pytest.approx(sind(0.0)) == 0.0
    assert pytest.approx(sind(90.0)) == 1.0
    assert pytest.approx(sind(180.0)) == 0.0
    assert pytest.approx(sind(270.0)) == -1.0


def test_tand():
    assert pytest.approx(tand(0.0)) == 0.0
    assert pytest.approx(tand(45.0)) == 1.0
    assert pytest.approx(tand(135.0)) == -1.0
    assert pytest.approx(tand(180.0)) == 0.0
    assert pytest.approx(tand(225.0)) == 1.0
    assert pytest.approx(tand(315.0)) == -1.0


def test_wrap_180():
    assert wrap_180(-180.0) == 180.0
    assert wrap_180(180.0) == 180.0
    assert wrap_180(-181.0) == 179.0
    assert wrap_180(-179.0) == -179.0
    assert wrap_180(179.0) == 179.0
    assert wrap_180(181.0) == -179.0


def test_wrap_360():
    assert wrap_360(0.0) == 0.0
    assert wrap_360(360.0) == 0.0
    assert wrap_360(-1.0) == 359.0
    assert wrap_360(1.0) == 1.0
    assert wrap_360(359.0) == 359.0
    assert wrap_360(361.0) == 1.0
