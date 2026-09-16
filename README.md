# FaceTrackVR

FaceTrackVR (FTVR) is a practical fork of
[EyeTrackVR](https://github.com/EyeTrackVR/EyeTrackVR) for running eye and
mouth tracking together in one application.

Compared with upstream EyeTrackVR, this fork currently adds:

- pupil-dilation tracking with EllSeg
- mouth tracking with custom and VRCFT-style post-processing
- a portable SRanipal-like runtime implementation for the Vive Facial Tracker

The initial target setup is:

- Bigscreen Beyond 2e eye cameras
- Vive Facial Tracker mouth camera
- VRChat output through the integrated VRCFT-compatible sender

FTVR is currently experimental and primarily developed to make this hardware
combination usable today. Changes are kept suitable for upstreaming where
possible.

## Current status

Eye tracking and the existing EyeTrackVR features remain available. The fork
adds an in-progress Linux Vive Facial Tracker capture path, mouth inference,
the compatible stateful postprocessor, and integrated face-expression output.

The direct Vive Facial Tracker camera backend is currently Linux-only. Other
EyeTrackVR functionality remains cross-platform to the same extent as the
upstream project.

## Experimental builds

Every commit on the `experimental` branch produces a rolling Linux x86_64
prerelease. The existing `experimental` release is replaced after a successful
build and focused test run, so its download always represents the newest
working commit.

These builds are intentionally prereleases. Check `SHA256SUMS` before running
the downloaded archive, and expect settings or behavior to change between
commits.

## Mouth model

FTVR does not include a mouth-tracking model. Select a compatible ONNX file in
the Mouth settings. One can be produced with [lip-tvm2onnx](https://github.com/mommysgoodpuppy/lip-tvm2onnx).

## Development

On Linux, install a source-backed development launcher with:

```bash
bash scripts/linux/install_source.sh
```

This creates a project-local uv environment and makes both the application-menu
entry and `facetrackvr` command run the current checkout directly. Source edits
therefore take effect on the next launch without rebuilding a release archive.

The inherited Poetry project metadata remains available in
[`pyproject.toml`](pyproject.toml).

The implementation plan is maintained separately while the initial fork is
being assembled. Setup and release instructions will be added before the first
usable release.

## Safety

EyeTrackVR supports custom infrared eye-tracking hardware. Infrared emitters
can cause irreversible eye injury when improperly designed or driven. Do not
bypass hardware safety measures, and do not use focused or unverified emitters.

## Origin and license

FaceTrackVR is a modified fork of EyeTrackVR. The upstream project and its
contributors remain credited in the source history and copyright notices.

Software is distributed under the included
[Babble Software Distribution License 1.0](LICENSE). Documentation inherited
from EyeTrackVR retains its applicable license. Third-party components retain
their own licenses and notices. The bundled EllSeg model is distributed under
the MIT license; its copyright and license text are included in
[`EyeTrackApp/Models/EllSeg_LICENSE.md`](EyeTrackApp/Models/EllSeg_LICENSE.md).
