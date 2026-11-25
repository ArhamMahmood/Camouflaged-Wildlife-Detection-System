# Final COD10K Test Performance

| Model         |   Test Images |   S-measure |   Mean E-measure |   Adaptive E-measure |   Weighted F-measure |      MAE |
|:--------------|--------------:|------------:|-----------------:|---------------------:|---------------------:|---------:|
| ResNet34-UNet |          4000 |    0.837567 |         0.849610 |             0.817374 |             0.235738 | 0.035727 |

Evaluation protocol: float probability predictions with `normalize=False`; all test images included.
