# herb-data-analysis

**AI Camp 4组成果**<br>

- 通过对《本草纲目》古籍的整理，利用大语言模型抽取中草药的 [名称,治疗病症,用法] 三个特征,制作得到了草药数据集 <br>
*herb_dataset.tar:* 该压缩包包含了887个中草药${herb_name}.json文件<br>
格式示例(阿芙蓉.json): <br>
```json
{
    "药物": "阿芙蓉",
    "治疗": [
        {
            "疾病": "久痢",
            "用法": "用阿芙蓉如小豆大小，每日空心服一次，温水化下。忌食葱蒜等物。"
        },
        {
            "疾病": "赤白痢下",
            "用法": "用阿芙蓉、木香、黄连、白术各一分，共研为末，加饭做成丸子，如小豆大。每服壮者一分，老幼半分，空心服，米汤送下。忌食酸物、生冷、油腻、茶、酒、面。又方；罂粟花未开时，外有两片青叶包着。花开即落，收取研末。每服一钱，米汤送下。赤痢用红花的包叶，白痢用白花的包叶。"
        }
    ]
}
```
- *embedding.py:* 读取herb_dataset 数据集文件，使用sentence transformer库，利用qwen3-embedding=8b模型(可替换为其它中文embedding模型)对每个中草药的"疾病"数据分别进行嵌入，再取平均得到中草药的嵌入向量shape=(n,4096),输出为json文件
- *visualization.py:* 读取中草药嵌入向量json数据，调用sklearn的t-SNE库对数据降到2维[shape=(n,2)]，再使用seaborn库生成等高线图，最后用plotly库生成可视化结果,输出html格式的图表界面<br>

预览输出结果(国内访问略慢)<br>
---->https://herb-data-visualization.edgeone.app/

