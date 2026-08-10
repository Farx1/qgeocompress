
from qgeocompress.valohai.export import run_export_if_accepted


def test_export_rejected_writes_rejection_manifest(tmp_path):
    gate = {"valohai_status": "rejected", "accepted_for_export": False, "reasons": ["mAP drop exceeds tolerance"]}
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    result = run_export_if_accepted(gate, model, output_dir=tmp_path / "out")
    assert result["manifest_name"] == "rejection_manifest.json"
    assert result["manifest"]["exported"] is False


def test_export_research_writes_research_manifest(tmp_path):
    gate = {"valohai_status": "accepted_for_research", "accepted_for_export": False}
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    result = run_export_if_accepted(gate, model, output_dir=tmp_path / "out")
    assert result["manifest_name"] == "research_manifest.json"
    assert result["manifest"]["exported"] is False


def test_export_accepted_copies_model(tmp_path):
    gate = {
        "valohai_status": "accepted_for_export",
        "accepted_for_export": True,
        "real_param_reduction_pct": 5.6,
    }
    model = tmp_path / "model.pt"
    model.write_bytes(b"x" * 1024)
    out = tmp_path / "out"
    result = run_export_if_accepted(gate, model, output_dir=out, try_onnx=False, try_torchscript=False)
    assert result["manifest_name"] == "export_manifest.json"
    assert result["manifest"]["exported"] is True
    assert (out / "export_model.pt").exists()
