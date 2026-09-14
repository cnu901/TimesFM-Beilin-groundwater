# TimesFM 地下水位异常识别项目：公共复现包

本包用于支撑论文《Covariate-Augmented TimesFM Foundation Model for Groundwater-Level Anomaly Detection and Earthquake-Association Analysis: A 13-Year Case Study at Beilin Well, Northeast China》。

研究定位是回溯性地下水异常识别和地震事件时间关联检验，不证明地震前兆，也不证明实时地震预测能力。探索性 30–90 天窗口为 11 个可评价事件中 6 个匹配（p = 0.002，FDR q = 0.008）；序列级检验为 2/3（p = 0.060）；回溯性 Molchan 技巧评分为 0.44。

原始监测文件以及含水位观测值的源数组受数据所有方共享规定约束，未放入本保守公共包。包内提供代码、CENC 和松原 M_S 目录、模型预测、异常段、统计结果、敏感性分析、图表和允许公开的源数据。详见 `DATA_NOTICE.md` 和 `docs/DATA_DICTIONARY_PUBLIC.md`。

运行方法和引用方式见 `README.md`、`docs/REPRODUCIBILITY_GUIDE_FINAL.md` 与 `CITATION.cff`。
