#文本分块器，输入是一个纯文本路径，输出是每块字符串
import os
import re




def read(input_data,encoding):
    if os.path.exists(input_data) and os.path.isfile(input_data):   #如果输入是文本路径
        try:
            with open(input_data,'r',encoding=encoding) as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(input_data,'r',encoding='gbk') as f:
                text = f.read()
    else:
        text = str(input_data)

    text = re.sub(r'\r\n', '\n', text)  # 同一换行符
    text = re.sub(r'\n{3,}', '\n\n', text)  # 多个空行合并成2个
    return text.strip()

def truncate_by_punctuation(text, n=50):
    """
    使用正则表达式查找字符串中距离开头和结尾最近的标点符号之间的文本

    参数:
    text: 输入字符串
    n: 从开头和结尾最多查找的字符数

    返回:
    两个标点符号之间的字符串，如果找不到则返回整个字符串
    """
    if not text:
        return text

    # 定义标点符号的正则表达式模式
    punctuation_pattern = r'[.,。!！?？;；:：,，]'

    # 查找开头部分的标点
    start_text = text[:n] if n is not None else text
    start_match = re.search(punctuation_pattern, start_text)
    start_pos = start_match.start() if start_match else None

    # 查找结尾部分的标点
    end_text = text[-n:] if n is not None else text
    end_matches = list(re.finditer(punctuation_pattern, end_text))
    end_pos = None

    if end_matches:
        end_match = end_matches[-1]
        end_pos = len(text) - len(end_text) + end_match.start()

    # 根据找到的位置提取文本
    if start_pos is not None and end_pos is not None and end_pos > start_pos:
        return text[start_pos + 1:end_pos]
    elif start_pos is not None:
        return text[start_pos + 1:]
    elif end_pos is not None:
        return text[:end_pos]
    else:
        return text


def chunk_by_paragraph(input_data,min_length=30,encoding: str = 'utf-8')->list[str]:
    """
    按段落分块
    :param input_data: 字符串
    :param min_length: 块的最低字数，小于则合并
    :param encoding:
    :return:
    """
    text=read(input_data,encoding)

    if not text:
        return []

    raw_paragraphs=re.split(r'\n\s*\n',text)
    raw_paragraphs=[para.strip() for para in raw_paragraphs if para.strip()]

    if not raw_paragraphs:
        return []

    chunks=[]
    current_chunk=""
    for paragraph in raw_paragraphs:
        if not current_chunk:
            current_chunk=paragraph
        elif len(current_chunk)<min_length:
            current_chunk+="\n\n"+paragraph
        else:
            chunks.append(current_chunk)
            current_chunk=paragraph
    if current_chunk:
        if len(current_chunk)<min_length and chunks:
            chunks[-1]+="\n\n"+current_chunk
        else:
            chunks.append(current_chunk)

    return [chunk for chunk in chunks if chunk.strip()]

def chunk_by_zh_chars(input_data,chunk_size=300,overlap=50,encoding: str='utf-8')->list[str]:
    """
    按字数分块
    :param input_data:
    :param chunk_size: 块的字数
    :param overlap: 前后重叠字数
    :param encoding:
    :return:
    """
    text = read(input_data, encoding)
    if not text:
        return []

    chunks=[]
    text_length=len(text)
    start=0
    while start<text_length:
        #计算当前块的结束位置
        end=start+chunk_size
        if end>=text_length:
            #最后一块
            chunk_text=text[start:text_length]
            if chunk_text.strip():
                chunks.append(chunk_text)
            break
        chunk_text=text[start:end]
        if chunk_text.strip():
            chunks.append(chunk_text)
        start=end-overlap
        if start<=0:
            start=0
            if chunks and len(chunks[0])==text_length:
                break

    if len(chunks)>1:
        last_chunk=chunks[-1]
        if len(last_chunk)<int(chunk_size/4):
            chunks.pop()
            if chunks:
                combined = chunks[-1] + last_chunk
                chunks[-1] = combined

    return [chunks[0].strip()]+ [truncate_by_punctuation(chunk) for chunk in chunks[1:-1] if chunk.strip()]+[chunks[-1].strip()]


def chunk_by_en_words(input_data,chunk_size,overlap,encoding: str='utf-8')->list[str]:
    words=input_data.split()
    chunks=[]
    for i in range(0,len(words),chunk_size-overlap):
        chunk=' '.join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

