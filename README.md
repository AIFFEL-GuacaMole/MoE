## 프로젝트 구조

```plaintext
├── MoE                             # MoE 구현
│   ├── data                        # Admet Group Datasets
│   ├── finetuning1D                # Chemberta 구현
│   ├── finetuning2D                # GIN 구현
│   ├── finetuning3D                # Uni-Mol 구현
│   ├── MoE                         # MoE 기법 파인튜닝 저장 디렉토리리
│   │   ├── cache                   # TDC Datasets GIN Features
│   │   ├── data                    # TDC Datasets
│   │   ├── MoE.py                  # 1D, 2D, 3D 모델을 이용한 MoE 구현 코드
│   │   ├── MoE2.py                 # 2D 모델을 이용한 MoE 구현 코드

```

## 주요 파일 설명

### **`MoE.py`**
- **역할:** ChemBERTa, GIN, Uni-Mol 모델을 사용한 MoE 학습 스크립트입니다.

### **`MoE2.py`**
- **역할:** 4개의 GIN 모델을 사용한 MoE 학습 스크립트입니다.

## Environment
Python 3.9
