# OpenCV AI Competition 2026: Requirements & Architectural Preservations

## 1. Competition Overview

- **Competition:** [OpenCV AI Competition 2026](https://opencv.org/) (Powered by AWS)
- **Project Name:** SignBridge
- **Domain:** Accessibility-focused sign language communication and interpretation system.

## 2. Mandatory Competition Requirements to Preserve

The following technical mandates must be preserved for all subsequent development phases:

1. **Substantive OpenCV 5 Image/Video Analysis:**
   - Computer vision must not be a trivial wrapper or superficial cosmetic layer.
   - Core visual features (keypoint tracking, spatio-temporal gesture dynamics, optical flow, pose/hand geometry analysis) must be substantively handled by the OpenCV 5 engine.
2. **Meaningful AWS Cloud Architecture:**
   - A substantive architectural component must execute on Amazon Web Services (AWS) — e.g., cloud-accelerated inference endpoints, asynchronous translation pipeline, or containerized ECS/Lambda orchestration.
   - *Current phase guard:* Do not provision AWS resources, configure live cloud infrastructure, or incur cloud costs during repository setup.
3. **Rigorous Technical Evaluation & Failure Modes:**
   - The final submission requires empirical evaluation detailing accuracy, latency, and throughput.
   - Crucially, documentation must include an honest analysis of **failure cases** (e.g., rapid occlusions, variable lighting, camera angles, non-standard dialect variations).
4. **Deliverables:**
   - Technical documentation and architecture reports.
   - Public open-source repository (Apache-2.0).
   - Working demonstration or live testable endpoint.

## 3. Current Project Status

- **Status:** Initial repository setup complete. No dependencies installed, no AWS resources provisioned.
- **Next Steps:** Architecture design, local OpenCV 5 pipeline prototyping, AWS architecture design, and benchmark data pipeline definition.
