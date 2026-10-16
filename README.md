# opencv-signbridge (SignBridge)

> **OpenCV AI Competition 2026** (Powered by AWS) &bull; Accessibility-Focused Sign Language Communication System

---

## 1. Project Overview

**SignBridge** is an accessibility-driven assistive technology application engineered to bridge the communication barrier between Deaf/Hard-of-Hearing individuals and non-signing communities.

The project pairs substantive **OpenCV 5** computer vision algorithms for real-time gesture analysis, hand pose extraction, and visual motion dynamics with scalable **Amazon Web Services (AWS)** cloud infrastructure for contextual translation, sequence modeling, and reliable delivery.

---

## 2. Competition Requirements & Preserved Constraints

> [!IMPORTANT]
> **Implementation Status: MVP Completion Phase**  
> The repository now includes a fully functional MVP demonstrating real-time gesture recognition using OpenCV 5, a FastAPI backend, and an AWS S3 integration fallback mechanism.

The following mandatory competition rules are strictly preserved for future development:
- **Substantive OpenCV 5 Core:** OpenCV 5 must perform substantive, real-time image and video processing (pose estimation, hand landmark localization, motion vectors), rather than functioning as a cosmetic wrapper.
- **Meaningful AWS Component:** A substantive component of the architecture must run on AWS (e.g., cloud-accelerated inference, translation sequence pipeline, or microservices).
- **Comprehensive Evaluation:** The final submission will document empirical benchmarks alongside an explicit evaluation of **failure cases** (illumination shifts, occlusion, motion blur).
- **Public Artifacts:** Open-source code under Apache-2.0, technical documentation, and a working demo or endpoint.

---

## 3. Strict Engineering & Anti-Fabrication Principles

- **No Fabricated Benchmarks:** All metrics, latency numbers, and accuracy figures will be empirically measured during development; none are fabricated or simulated.
- **No Committed Secrets:** AWS access keys, secret credentials, and private environment files are excluded via `.gitignore` and must never be committed.
- **Independent Project:** Autonomous repository with decoupled version control, tests, documentation, and dependencies.

---

## 4. Repository Structure

```
opencv-signbridge/
├── LICENSE                          # Apache-2.0 License
├── README.md                        # Project documentation & status
├── .gitignore                       # Environment, secret, build & media guards
├── docs/                            # Competition requirements & architecture
│   └── competition_requirements.md  # Preserved competition requirements
└── tests/                           # Computer vision & integration tests (planned)
```

---

## 5. Setup and Execution

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate Data and Train Model:**
   ```bash
   python scripts/generate_data.py
   python scripts/train.py
   ```
3. **Run Demo Script:**
   ```bash
   PYTHONPATH=. python scripts/demo.py
   ```
4. **Run Web Server:**
   ```bash
   PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
5. **Run Tests:**
   ```bash
   PYTHONPATH=. pytest tests/
   ```

---

## 6. Architecture & Results

- **Vision Pipeline:** Uses OpenCV 5 for color space conversion (YCrCb), morphological noise removal, contour detection, and convex hull/defects calculation to extract hand geometry features.
- **Model:** A Scikit-Learn `RandomForestClassifier` trained on synthetic geometric features.
- **AWS Integration:** Uses `boto3` to upload evaluation JSON reports to S3, with a graceful local fallback mechanism if AWS credentials are not configured.
- **Evaluation:** Evaluated on synthetic isolated gesture data (`HELLO`, `PEACE`, `OKAY`). F1 score on synthetic data is 1.0 (Note: Subject to real-world evaluation later).
- **Limitations:** This MVP recognizes a bounded set of isolated gestures and does not translate unrestricted continuous sign language.

## 7. License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
