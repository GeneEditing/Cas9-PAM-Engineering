# 第三方模型与运行接口

本项目只整合公开发布的模型进行推理，**不包含模型权重，不进行训练或微调**。模型需由使用者从上游项目单独获取，并遵守各自的许可、引用和下载条款。

| 模型 | 任务 | 本仓库调用 | 资源目录 |
| --- | --- | --- | --- |
| [PAMmla](https://github.com/RachelSilverstein/PAMmla)，seed 0/2/3 | 六位点 PAM 预测 | TensorFlow/Keras 加载三个 SavedModel 并集成平均；[推理入口](../src/steps/01_pammla/batch_predict.py) | `resources/PAMmla/220924_NN_rand_seed{0,2,3}_ROS/` |
| [ProGen2-base](https://github.com/salesforce/progen/tree/main/progen2) | 全长序列似然评分 | PyTorch 加载 ProGen2 模型实现与 checkpoint；[推理入口](../src/steps/03_progen2/score_progen2.py) | `resources/progen2/` |
| [ESM-C-600M](https://huggingface.co/biohub/ESMC-600M) | 突变上下文评分 | Transformers 本地加载 `AutoTokenizer` 和 `AutoModelForMaskedLM`；[推理入口](../src/steps/04_esmc/score_esmc.py) | `resources/esmc_models/ESMC-600M/` |
| [ColabFold 1.6.2](https://github.com/sokrypton/ColabFold/releases/tag/v1.6.2) / AlphaFold2-ptm | 结构预测 | 调用环境中的 `colabfold_batch` CLI；参数获取见 [官方说明](https://github.com/sokrypton/ColabFold/wiki/Running-ColabFold-in-Docker) | `resources/colabfold/cache/` |

ProGen2 和 ESM-C 是本流程的蛋白质语言模型；PAMmla 是 PAM 预测器，ColabFold 是结构预测工具。推荐从仓库根目录使用 `bash run.sh run` 统一调度，而不是绕过流程单独调用步骤脚本。环境定义见 `requirements/*.environment.yml`。

## 权重与校验

从上述上游仓库、Hugging Face 模型页和 ColabFold 官方文档获取相符版本的资源，放入对应目录。ProGen2 许可见其上游说明和仓库 `LICENSE.txt`；PAMmla 许可见上游 `LICENSE`；ESM-C 许可见模型页及第三方声明；ColabFold 代码与 AlphaFold 参数需分别核对许可。

`WEIGHT_MANIFEST.tsv` 记录历史使用的 30 个文件的路径、大小和 SHA256（约 5.32 GiB），并不包含权重内容，也不是官方镜像。手动下载后执行 `python3 models/download_weights.py check` 校验。只有在设置 `CAS9_WEIGHT_BASE_URL` 为你有权使用的可信镜像时，`bash run.sh weights` 才会尝试补齐缺失文件；仓库不配置历史私有镜像地址。文件校验不通过时不要替换清单或静默使用其他 checkpoint。

历史记录使用 ESM-C 缓存 revision `a7e82012c83126b9eedb055fea9fa84b6c02f094`、PAMmla seed 0/2/3、ProGen2-base 和五份 AlphaFold2-ptm 参数。上游资源或许可可能变化，使用前请复核模型卡及许可。
