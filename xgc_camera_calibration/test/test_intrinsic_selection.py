import copy
import tempfile
import unittest
from pathlib import Path
import yaml
from xgc_camera_calibration.intrinsic_selection import resolve_intrinsic
from xgc_camera_calibration.solver import CalibrationError


class IntrinsicSelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "camera's calibrated files"
        self.document = {"schema":"xgc2.camera.intrinsic.v1", "camera_name":"usb_cam",
            "calibration_mode":"phy", "image_width":3840,"image_height":2160,
            "camera_matrix":{"data":[1900,0,1920,0,1900,1080,0,0,1]},
            "distortion_coefficients":{"data":[0.01,0,0,0,0]},
            "metadata":{"stability_assessment":{"passed":True}}}

    def save(self, stamp, document=None, mode="phy"):
        path=self.root/mode/"usb_cam"/("intrinsics-20260907T"+stamp+".000000Z.yaml")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(document or self.document))
        return path

    def resolve(self, explicit="", mode="phy", policy="latest"):
        return resolve_intrinsic(str(self.root),mode,"usb_cam",str(explicit),(3840,2160),policy)

    def test_empty_history_is_read_only_default(self):
        result=self.resolve()
        self.assertEqual(result["file"],"")
        self.assertFalse(self.root.exists())

    def test_selects_latest_once_and_keeps_explicit_old_revision(self):
        old=self.save("010000");new=self.save("020000")
        original={path:path.read_bytes() for path in (old,new)}
        result=self.resolve()
        self.assertEqual(result["file"],str(new))
        self.assertEqual(self.resolve(old)["file"],str(old))
        self.save("030000")
        self.assertEqual(result["file"],str(new))
        for path,data in original.items():self.assertEqual(path.read_bytes(),data)
        self.assertEqual(self.resolve(policy="default")["file"],"")

    def test_other_modes_cameras_sizes_and_failed_quality_are_not_auto_selected(self):
        valid=self.save("010000")
        variants=[("020000",{"calibration_mode":"sim"}),
                  ("030000",{"camera_name":"other"}),
                  ("040000",{"image_width":1920}),
                  ("050000",{"metadata":{"stability_assessment":{"passed":False}}})]
        for stamp,fields in variants:
            document=copy.deepcopy(self.document);document.update(fields);self.save(stamp,document)
        result=self.resolve()
        self.assertEqual(result["file"],str(valid))
        self.assertEqual(len(result["rejected"]),4)
        self.assertEqual(self.resolve(mode="sim")["file"],"")

    def test_explicit_bad_file_fails_without_fallback(self):
        self.save("010000")
        wrong=copy.deepcopy(self.document);wrong["calibration_mode"]="sim"
        path=self.save("020000",wrong)
        with self.assertRaises(CalibrationError):self.resolve(path)
        with self.assertRaises(ValueError):self.resolve(self.root/"sim"/path.name)

    def test_corrupt_latest_is_reported_and_legacy_mode_is_identified(self):
        legacy=copy.deepcopy(self.document);del legacy["calibration_mode"]
        path=self.save("010000",legacy)
        bad=self.save("020000");bad.write_text("invalid: [")
        result=self.resolve()
        self.assertEqual(result["file"],str(path))
        self.assertEqual(result["mode_identity"],"legacy-directory")
        self.assertEqual(len(result["rejected"]),1)

    def test_symlink_cannot_cross_mode(self):
        sim=copy.deepcopy(self.document);sim["calibration_mode"]="sim"
        path=self.save("010000",sim,mode="sim")
        link=self.root/"phy"/"usb_cam"/path.name
        link.parent.mkdir(parents=True);link.symlink_to(path)
        with self.assertRaises(ValueError):self.resolve(link)
