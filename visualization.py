import json
import os
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_validate_data(embedding_path, dataset_path):
    """加载并验证嵌入数据和病症信息"""
    with open(embedding_path, 'r', encoding='utf-8') as f:
        embedding_data = json.load(f)
    
    herbs = []
    embeddings = []
    
    for herb, vec in embedding_data.items():
        if isinstance(vec, list) and len(vec) == 4096:
            herbs.append(herb)
            embeddings.append(vec)
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    symptoms = {}
    for filename in os.listdir(dataset_path):
        if filename.endswith('.json'):
            filepath = os.path.join(dataset_path, filename)
            print(f"正在处理文件: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    symptom_data = json.load(f)
                herb_name = symptom_data.get("药物")
                treatments = symptom_data.get("治疗", [])
                treatment_info = "<br>".join(
                    [f"疾病: {t['疾病']}<br>用法: {t['用法']}" for t in treatments]
                )
                symptoms[herb_name] = treatment_info
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {str(e)}")
    
    return herbs, embeddings, symptoms

def reduce_dimensions(embeddings):
    """使用t-SNE降维"""
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1500,
        init='pca',
        random_state=42,
        learning_rate='auto'
    )
    return tsne.fit_transform(embeddings)

def create_density_plot(coords):
    """使用matplotlib和seaborn创建密度等高线图"""
    plt.figure(figsize=(10, 8))
    
    sns.kdeplot(
        x=coords[:, 0], 
        y=coords[:, 1], 
        cmap="Blues", 
        fill=True, 
        alpha=0.5, 
        levels=10
    )
    
    sns.kdeplot(
        x=coords[:, 0], 
        y=coords[:, 1], 
        color='darkblue', 
        linewidths=0.5, 
        levels=5
    )
    
    plt.xlabel('t-SNE 维度1', fontsize=12)
    plt.ylabel('t-SNE 维度2', fontsize=12)
    plt.title('中草药分布密度图', fontsize=14)
    
    plt.savefig('herb_density_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_interactive_plot_with_contours(herbs, coords, symptoms=None):
    """创建带有密度等高线的交互式可视化图表"""
    df = pd.DataFrame({
        '草药名称': herbs,
        'x': coords[:, 0],
        'y': coords[:, 1]
    })
    
    if symptoms:
        df['病症信息'] = df['草药名称'].map(lambda x: symptoms.get(x, "无病症记录"))
    
    # 创建基础散点图
    fig = go.Figure()
    
    # 添加密度等高线
    fig.add_trace(go.Histogram2dContour(
        x=coords[:, 0],
        y=coords[:, 1],
        colorscale='Blues',
        reversescale=False,
        showscale=False,
        contours=dict(
            coloring='fill',
            showlines=False
        ),
        opacity=0.5
    ))
    
    # 添加散点
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=11,
            color='lightblue',
            line=dict(width=0.5, color='DarkSlateGrey'),
            opacity=0.8
        ),
        text=df['草药名称'],
        customdata=df[['草药名称', '病症信息']] if symptoms else df[['草药名称']],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br><br>" +
            "维度1: %{x:.2f}<br>" +
            "维度2: %{y:.2f}<br>" +
            "%{customdata[1]}" +
            "<extra></extra>"
        ) if symptoms else (
            "<b>%{customdata[0]}</b><br><br>" +
            "维度1: %{x:.2f}<br>" +
            "维度2: %{y:.2f}<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title="基于《本草纲目》的中草药治疗病症相似度可视化（含密度分布）图像",
        title_font=dict(size=24),
        xaxis_title='t-SNE 维度1',
        yaxis_title='t-SNE 维度2',
        xaxis=dict(fixedrange=False,
                   range=[-18, 18],  # 固定X轴范围
                   autorange=False,),  # 允许X轴缩放（默认就是False）
        yaxis=dict(range=[-22, 22]),
        width=1200,
        height=800,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Microsoft YaHei"
        ),
        font=dict(family="Microsoft YaHei"),
        plot_bgcolor='rgba(240,240,240,0.85)',
        dragmode='pan'
    )
    
    return fig

def add_description(fig, description_text):
    """在图表右侧添加说明文本"""
    fig.add_annotation(
        text=description_text,
        xref="paper", yref="paper",
        x=1.1, y=0,
        showarrow=False,
        align="left",
        font=dict(size=14, family="Microsoft YaHei"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )

def main(embedding_path, dataset_path):
    """主函数"""
    print("正在加载数据...")
    herbs, embeddings, symptoms = load_and_validate_data(embedding_path, dataset_path)
    print(f"成功加载 {len(herbs)} 种草药数据")
    
    print("开始降维处理...")
    reduced_embeddings = reduce_dimensions(embeddings)
    
    print("生成密度等高线图...")
    create_density_plot(reduced_embeddings)
    
    print("生成交互式可视化图表...")
    fig = create_interactive_plot_with_contours(herbs, reduced_embeddings, symptoms)
    
    description_text = """
    这是一个中草药相似度可视化图表。
    每个点代表一种草药，点之间的距离表示它们之间的相似度。
    等高线表示草药分布的密度。
    悬停在点上可以查看草药的名称和相关病症信息。
    
    """
    add_description(fig, description_text)
    
    output_file = "herb_visualization.html"
    fig.write_html(output_file)
    print(f"可视化已保存为 {output_file}")
    
    # fig.show(renderer="browser")

if __name__ == "__main__":
    embedding_file = "entity_embeddings.json" #change to your entity_embedddings.json path
    dataset_path = "herb_dataset" #change to your herb_dataset path
    
    try:
        main(embedding_file, dataset_path)
    except Exception as e:
        print(f"程序出错: {str(e)}")
