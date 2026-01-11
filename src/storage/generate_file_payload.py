#获取一个文件夹下的所有文件,并生成元数据
import os
from pathlib import Path


class PayloadBuilder:
    """扫描文件并生成元数据"""
    def __init__(self):
        self.file_type_mapping={"document":["pdf","docx"],    #文档类型，需要特定软件打开
                                "text":["txt","json","log","md"],   #纯文本
                                "image":["jpg","jpeg","png","bmp"], #常见图片类型
                                }

        self.ext_map_category=self._ext_map_category()


    def _ext_map_category(self):
        extension_to_category = {}

        for category, extensions in self.file_type_mapping.items():
            for ext in extensions:
                extension_to_category[ext] = category

        return extension_to_category

    def search_files(self,folder_path):
        folder_path = Path(folder_path).resolve()
        payloads=[]
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()[1:]  # 去掉点，转小写
                if ext in self.ext_map_category:
                    payloads.append({
                        "path": str(file_path.resolve()),
                        "name": file_path.name,
                        "extension": ext,
                        "category": self.ext_map_category[ext]
                    })
        return payloads

    def generate_payload(self,file_path):
        file_path = Path(file_path)
        if file_path.is_file():
            ext = file_path.suffix.lower()[1:]  # 去掉点，转小写
            if ext in self.ext_map_category:
                return {
                    "path": str(file_path.resolve()),
                    "name": file_path.name,
                    "extension": ext,
                    "category": self.ext_map_category[ext]
                }
            else:
                print("该文件格式不支持")
        else:
            print("该文件不存在！！！")





