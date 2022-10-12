
# **6DRepNet ONNX**

공식 저자의 깃헙입니다. 
[Official Repo](https://github.com/thohemp/6DRepNet) 해당 깃헙의 코드를 기반으로 ONNX 컨버팅을 수행했습니다.


## Update
- **```2022/10/12```** : init

## Dirs
```
    ├── README.md               
    ├── src
    │   ├── backbone             
    │   │   ├── repvgg.py        <- backbone network
    │   ├── utils.py             <- utils
    │   ├── model.py             <- 6drepnet network
    │   │
    │   ├── convert_onnx.py      <- convert to onnx
    │   └── demo.py              <- demo
    ├── models
```

## Pre-train Models
ONNX 컨버팅 또는 DEMO 실행을 위해 pre-trained model을 다운로드해야 합니다. 다운받은 파일은 `models` 폴더에 적재해야합니다.


**6DRepNet_300W_LP_BIWI.pth** `pre_train backbone` [다운로드 링크](https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_)
<br>
**RepVGG-B1g2-train.pth** `pre_train 6drepnet` [다운로드 링크](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq)
<br>
**sixdrepnet.onnx** `onnx 6drepnet` [다운로드 링크](https://drive.google.com/file/d/1mIle66NaI_Iho3Cbyvl-wgyVvMPeJwo4/view?usp=sharing)
<br>

## Convert ONNX
`torch` 라이브러리가 필요합니다. `--name`을 통해 컨버팅 된 파일의 이름을 정의할 수 있습니다.
```
python3 convert_onnx.py --name want_to_use_name.onnx
```

## DEMO
`mediapipe`, `onnxruntime`, `opencv` 라이브러리가 필요합니다.<br>
`--draw_bbox 1`를 통해 bounding box를 시각화 할 수 있습니다.<br>
`--draw_cube 1`를 통해 headpose cube box를 시각화 할 수 있습니다.<br>
`--draw_line 1`를 통해 headpose line을 시각화 할 수 있습니다.<br>
```
python3 demo.py --draw_bbox 1 --draw_line 1
```