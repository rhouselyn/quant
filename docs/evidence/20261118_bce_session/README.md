# 本次 BCE 会话的轻量证据

日期2026-11-18，Asia/Shanghai。主入口为[会话归档](../../research_20261118_bce_session_archive_results.md)。

本目录保留76份既有关键文件的逐字节复制件，合计783760字节。文件清单、来源、长度和SHA256见[manifest](manifest.json)。最终审计的39个原引用也已重新校验。新增本README属于说明文件，不计入原始证据76份。

- 当前有效数值见[295周最终66行指标](batches/bce_final_audit_v1/final_metrics.csv)、[配对区间](batches/bce_final_audit_v1/calendar_paired.csv)、[完整信号](batches/bce_final_audit_v1/complete_calendar_signals.csv)和[验收](batches/bce_final_audit_v1/verification.json)。
- 单因素核对见[Q1覆盖](batches/p6_q1_features_v1/feature_coverage.csv)、[监督指纹](batches/p6_q1_features_v1/training_identity.json)、[Q1验收](batches/p6_q1_features_v1/verification.json)和[预处理验收](batches/p6_preprocessing_v1/verification.json)。
- 其它批次中的283周结果保留为历史条件对照。76份源文件中的outputs目录映射为本目录batches目录，代码内容未改路径。

源码复制件用于审阅及重建原脚本；依赖仍需按原项目路径恢复，运行时输出应置于隔离目录以保留旧证据。大体量原行情、向量、模型权重及Parquet账本保留在原data/outputs，本目录没有完整独立运行所需的全量数据。原始资产仅本机，未核实异地备份。所有复制件可按manifest重新检查；本次未自动Git提交或推送。
