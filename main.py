from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
def check_vectors(vectors):
    if isinstance(vectors, np.ndarray) and np.issubdtype(vectors.dtype, np.floating):
        return True
    else:
        return False
def insert_vectors_to_milvus(vectors,website_urls):
    # 连接到 Milvus
    connections.connect("default", host='192.168.2.18', port='19530')

    b = check_vectors(vectors)
    # 创建集合
    collection = Collection(name="test")

    insert_data = [
        [vector.tolist() for vector in vectors],
        website_urls
    ]


    mr = collection.insert(column_names=['vector', 'url'], data=insert_data)

    collection.flush()

    # 获取插入向量的ID
    ids = mr.primary_keys
    print(f"插入的向量ID: {ids}")

    # 关闭 Milvus 连接
    connections.disconnect("default")
    return ids



def fetch_website_structure(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 将抛出异常，如果请求不成功
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"请求错误: {url}，错误信息：{e}")
        return None


def extract_tag_paths(soup, tag, path=""):
    if not soup or tag.parent is None:
        return path
    if tag.parent.name:
        path = extract_tag_paths(soup, tag.parent, path)
    path = f"{path}/{tag.name}" if path else tag.name
    return path


def vectorize_structure(tag_paths):
    # 初始化一个512维的零向量
    structure_vector = [0] * 512

    # 对于每个标签路径，使用散列函数来确定其在向量中的位置
    for path in tag_paths:
        # 使用hash函数取模得到一个0-511之间的索引
        index = hash(path) % 512
        # 在相应的位置上增加计数
        structure_vector[index] += 1

    return structure_vector


def plot_vectors(norm_vectors, labels, scale_factor=50):
    plt.figure()
    for i, (norm_vector, label) in enumerate(zip(norm_vectors, labels)):
        x, y = norm_vector[0] * scale_factor, norm_vector[1] * scale_factor
        plt.plot(x, y, 'o', label=f'{label}')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Normalized Website Structure Vectors')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


# 主程序
if __name__ == "__main__":
    website_urls = ['https://carefully.top', 'https://www.modamias.com',"https://www.jessielike.com/","https://www.carefully.top"]
    vectors = []

    # 获取所有网站的结构
    for url in website_urls:
        soup = fetch_website_structure(url)
        if soup:
            tag_paths = [extract_tag_paths(soup, tag) for tag in soup.find_all()]
            vector = vectorize_structure(tag_paths)
            vectors.append(vector)

        # 归一化向量并计算距离，加入零向量检查
    norm_vectors = np.array(
            [vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector for vector in vectors])
    insert_vectors_to_milvus(norm_vectors,website_urls)

    distances = euclidean_distances(norm_vectors)

    # 绘制归一化向量
    # plot_vectors(norm_vectors, [f'Website {i + 1}' for i in range(len(website_urls))])

    # 输出每个网站的坐标和它们之间的欧氏距离
    for i, norm_vector in enumerate(norm_vectors):
        x, y = norm_vector[0], norm_vector[1]
        print(f"网站{i + 1}的坐标: ({x:.2f}, {y:.2f})")
