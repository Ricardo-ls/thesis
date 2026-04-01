from pathlib import Path

# 从当前项目目录开始找文件
project_dir = Path(".")

print("当前项目目录：", project_dir.resolve())
print("\n项目里的文件和文件夹如下：\n")

for p in project_dir.rglob("*"):
    print(p)