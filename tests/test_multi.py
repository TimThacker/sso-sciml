"""Several converters on one bus."""

import numpy as np
import pytest

from sso_sciml import multi
from sso_sciml import vectorfit as vf
from sso_sciml.data import c1
from sso_sciml.physics.converter import Converter

COL = {name: j for j, name in enumerate(multi.COLUMNS)}


def unit(theta, follow, share=1.0, regulate=True, p=None):
    th = dict(zip(c1.NAMES, map(float, theta), strict=True))
    model = Converter(c1.controls(th, follow))
    return multi.Unit(model, th["p"] if p is None else p, share, regulate), c1.network(th)


@pytest.mark.parametrize(("family", "follow"), [("pll", 0.0), ("follow", 1.0)])
def test_one_unit_is_the_converter_of_dataset_c1(family, follow):
    for theta in c1.sample(4, family):
        model, x0, p, net = c1.operating_point(theta, follow)
        u, _ = unit(theta, follow, regulate=not follow)
        v, cur, refs = multi.operating_point([u], net)
        scans = multi.plant([u], v, cur, refs)
        assert multi.growth(scans, [1.0], net) == pytest.approx(
            c1.growth(model, x0, p)[0], abs=1e-7
        )


def test_two_halves_of_one_converter_add_only_the_modes_between_them():
    theta = c1.sample(4, "pll")[1]
    whole, net = unit(theta, 0.0)
    v, cur, refs = multi.operating_point([whole], net)
    one = multi.plant([whole], v, cur, refs)
    zero = np.zeros((2, 2))
    modes_one = vf.closed_loop_modes(*multi.side_by_side(one, [1.0]), zero, net)
    # Two identical units at the same operating point. Moving together they are the one
    # converter. Moving against each other they draw no net current, the bus does not
    # move, and each is the converter alone on an ideal voltage.
    half = multi.Unit(whole.model, whole.p_src, 0.5)
    two = multi.plant([half, half], v, [cur[0], cur[0]], [refs[0], refs[0]])
    modes_two = vf.closed_loop_modes(*multi.side_by_side(two, [0.5, 0.5]), zero, net)
    assert modes_two.size == modes_one.size + one[0].poles.size
    for value in (*modes_one, *one[0].poles):
        assert np.abs(modes_two - value).min() < 1e-6 * max(1.0, abs(value))


def test_bus_voltage_is_held_by_the_regulating_unit():
    theta = c1.sample(4, "pll")
    src, net = unit(theta[0], 0.0, share=0.6, p=0.7)
    load, _ = unit(theta[1], 0.0, share=0.4, regulate=False, p=-0.5)
    v, cur, refs = multi.operating_point([src, load], net)
    assert abs(v) == pytest.approx(1.0, abs=1e-9)
    assert refs[1] == 0.0
    # The network sees the weighted sum of the currents.
    z = multi._z60(net)
    assert v == pytest.approx(1.0 + z * (0.6 * cur[0] + 0.4 * cur[1]), abs=1e-9)


def test_a_unit_without_a_pll_cannot_regulate():
    load, net = unit(c1.sample(4, "follow")[0], 1.0, regulate=True)
    with pytest.raises(ValueError):
        multi.operating_point([load], net)


def test_case_row_matches_its_columns_and_sampling_repeats():
    jobs = multi.sample(4, "follow")
    assert all(
        np.array_equal(a[1], b[1]) for a, b in zip(jobs, multi.sample(4, "follow"), strict=True)
    )
    row = multi.case(jobs[0])
    assert row.shape == (len(multi.COLUMNS),)
    share = row[COL["share"]]
    assert row[COL["p_net"]] == pytest.approx(
        share * row[COL["p_source"]] + (1 - share) * row[COL["p_load"]]
    )
