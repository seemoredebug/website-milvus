import requests
from bs4 import BeautifulSoup
import numpy as np
from pymilvus import (
    connections,
    Collection
)


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


def fetch_website_structure(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

def extract_tag_paths(soup, tag):
    # 根据tag生成路径，这个实现依赖于具体需求
    # 例如，可以使用tag的名字或类别等作为路径
    path = tag.name  # 这是一个简化示例，仅使用tag的名称
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

def search_similar_websites(query_url, top_k, collection_name='test'):
    # 连接到 Milvus 实例
    connections.connect("default", host='192.168.2.18', port='19530')

    # 1. 抓取给定网站的结构
    query_soup = fetch_website_structure(query_url)

    all_tag_paths = set( [extract_tag_paths(query_soup, tag) for tag in query_soup.find_all()])

    # 2. 将结构转化为向量
    if query_soup:
        query_tag_paths = [extract_tag_paths(query_soup, tag) for tag in query_soup.find_all()]
        # 注意：这里我们假设 all_tag_paths 已经在之前被创建并存储了所有标签路径
        query_vector = vectorize_structure(query_tag_paths)
        #
        # if len(query_vector) < 430:
        #     # 创建一个零向量来填充差额
        #     padding = np.zeros(430 - len(query_vector))
        #     # 将查询向量与零填充向量连接起来
        #     query_vector_padded = np.concatenate((query_vector, padding))
        # else:
        #     # 如果查询向量的长度正好或超出预期，可以直接使用或截断
        #     query_vector_padded = query_vector[:430]
        # 3. 标准化向量
        query_norm_vector = query_vector / np.linalg.norm(query_vector)


        # 获取集合的引用
        collection = Collection(name=collection_name)

        # 4. 使用 Milvus 执行向量搜索
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        search_results = collection.search(
            data=[query_norm_vector.tolist()],
            anns_field='vector',
            param=search_params,
            limit=top_k,
            output_fields=['url'],
            expr=None
        )

        # 断开所有连接
        connections.disconnect(alias='default')

        if search_results:
            print(f"Top {top_k} similar websites to {query_url} are:")
            for hits in search_results:
                for hit in hits:
                    print(f"URL: {hit.entity.get('url')}, Similarity: {hit.score}")
        else:
            print("No similar websites found.")

# 使用示例
if __name__ == "__main__":
    search_similar_websites('https://modamias.com/', top_k=5)
