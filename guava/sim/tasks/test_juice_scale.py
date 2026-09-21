import xml.etree.ElementTree as ET
import numpy as np
import pytest
from guava.sim.tasks import apple_juice_order as task


def test_juice_matches_upstream_scale_and_placement_bounds():
    obj = task.JuiceObject('juice')
    for mesh in obj.asset.findall('mesh'):
        assert np.allclose(np.fromstring(mesh.get('scale'), sep=' '), [.4]*3)
    bbox = obj.get_obj().find('.//geom[@name="juice_reg_bbox"]')
    assert np.allclose(np.fromstring(bbox.get('size'), sep=' '), [.02114, .02114, .05850], atol=1e-5)
    assert obj.bottom_offset[2] == -.05850
    assert obj.top_offset[2] == .05850
    assert obj.horizontal_radius == .02114


@pytest.mark.parametrize('source_scale', [.4, 1.0])
def test_asset_scale_and_bounds_used_without_override(tmp_path, monkeypatch, source_scale):
    source = task._ROBOCASA_DIR/'Juice001/model.xml'
    tree = ET.parse(source)
    for node in tree.iter():
        if 'file' in node.attrib:
            node.set('file', str(source.parent/node.get('file')))
    for mesh in tree.findall('./asset/mesh'):
        mesh.set('scale', ' '.join([str(source_scale)] * 3))
    bbox = tree.find('.//geom[@name="reg_bbox"]')
    bbox.set('size', ' '.join(str(v*source_scale/.4) for v in [.02114, .02114, .05850]))
    dest = tmp_path/'Juice001/model.xml'
    dest.parent.mkdir()
    tree.write(dest)
    monkeypatch.setattr(task, '_ROBOCASA_DIR', tmp_path)
    obj = task.JuiceObject('juice')
    assert all(np.allclose(np.fromstring(m.get('scale'), sep=' '), [source_scale]*3) for m in obj.asset.findall('mesh'))
    bbox = obj.get_obj().find('.//geom[@name="juice_reg_bbox"]')
    expected = np.array([.02114, .02114, .05850]) * source_scale / .4
    assert np.allclose(np.fromstring(bbox.get('size'), sep=' '), expected)
    assert obj.bottom_offset[2] == pytest.approx(-expected[2])
    assert obj.top_offset[2] == pytest.approx(expected[2])
    assert obj.horizontal_radius == pytest.approx(expected[0])
