#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCoder Cloud Setup — one script to rule them all.
Run: python setup_cloud.py && python train.py --config small --steps 100000 --data chinese_data

Generates ~500K Chinese conversation turns + curated Python code + Chinese tokenizer.
"""

import os, sys, json, random, subprocess
from pathlib import Path

DATA_DIR = Path("chinese_data")
DATA_DIR.mkdir(exist_ok=True)
CHAT_DIR = DATA_DIR / "chats"
CHAT_DIR.mkdir(exist_ok=True)

random.seed(42)

print("=" * 60)
print("  Generating MASSIVE Chinese training data...")
print("=" * 60)

# =========================================================================
# HELPER
# =========================================================================
def make_qa(pairs, repeat=2000):
    """Generate Q&A pairs from (question, answer) tuples."""
    out = []
    for q, a in pairs:
        for _ in range(repeat):
            out.append(f"用户: {q}\n助手: {a}\n")
    return out

def make_multi(turns_list, repeat=1000):
    """Generate multi-turn conversations."""
    out = []
    for turns in turns_list:
        lines = []
        for i in range(0, len(turns), 2):
            lines.append(f"用户: {turns[i]}")
            lines.append(f"助手: {turns[i+1]}")
        block = "\n".join(lines) + "\n"
        for _ in range(repeat):
            out.append(block)
    return out

# =========================================================================
# PART 1: MASSIVE Chinese Conversations (500K+ turns)
# =========================================================================
print("\n[1/4] Generating Chinese conversations (500K+)...")

all_chats = []

# --- 1A: Daily chat (150 topics x 800 = 120K) ---
daily_chat = [
    ("你好", "你好！有什么我可以帮你的吗？"),
    ("你好呀", "嗨！今天过得怎么样？"),
    ("早上好", "早上好！今天有什么计划吗？"),
    ("早安", "早安！新的一天开始了！"),
    ("晚上好", "晚上好！今天过得开心吗？"),
    ("晚安", "晚安，做个好梦！"),
    ("好困啊", "早点休息吧，睡眠很重要！"),
    ("我睡不着", "数羊试试？或者听听轻音乐。"),
    ("睡醒了", "早上好！睡得好吗？"),
    ("今天天气真好", "是啊，适合出去走走！"),
    ("下雨了", "记得带伞哦！"),
    ("下雪了", "好美！注意保暖！"),
    ("好冷啊", "多穿点衣服，别感冒了。"),
    ("好热啊", "开空调或者喝点冷饮吧。"),
    ("今天心情不好", "怎么了？跟我说说？"),
    ("心情真好", "开心就好！有什么好事？"),
    ("我太难了", "别灰心，困难都是暂时的！"),
    ("好无聊", "找点事做吧，看个电影或者读本书？"),
    ("好开心", "分享下快乐！"),
    ("加油", "一起加油！"),
    ("谢谢", "不客气！很高兴能帮到你。"),
    ("非常感谢", "不用谢，能帮到你是我的荣幸！"),
    ("对不起", "没关系，不必在意！"),
    ("没事", "那就好！"),
    ("好的", "嗯嗯！"),
    ("好的吧", "好的吧~"),
    ("明白了", "明白就好！"),
    ("不懂", "哪里不懂？我帮你解释。"),
    ("然后呢", "接下来嘛..."),
    ("真的吗", "当然是真的！"),
    ("不会吧", "真的！我没骗你。"),
    ("厉害了", "过奖啦！"),
    ("你好棒", "谢谢夸奖！我会继续努力的。"),
    ("真聪明", "谢谢！跟你学的！"),
    ("再见", "再见！下次聊！"),
    ("拜拜", "拜拜！"),
    ("明天见", "明天见！"),
    ("回头聊", "好的，回头见！"),
    ("在吗", "在的！有什么需要帮忙的吗？"),
    ("在干嘛", "在想怎么帮你呢！"),
    ("忙不忙", "不忙，你说！"),
    ("吃饭了吗", "吃了！你呢？"),
    ("今天吃什么", "好吃的！想吃中餐还是西餐？"),
    ("好饿啊", "快去吃饭吧！"),
    ("好饱", "吃多了就散散步。"),
    ("渴了", "多喝水！对身体好。"),
    ("累了", "休息一下吧，别太拼了。"),
    ("生病了", "多喝热水，好好休息，严重的话去看医生。"),
    ("头疼", "是不是没睡好？休息一下看看。"),
    ("感冒了", "多穿点，按时吃药，好好休息。"),
    ("生日快乐", "生日快乐！愿你每天都开心！"),
    ("新年快乐", "新年快乐！万事如意！"),
    ("圣诞快乐", "圣诞快乐！"),
    ("节日快乐", "节日快乐！"),
    ("周末愉快", "周末愉快！有什么计划吗？"),
    ("周一好", "周一好！新的一周加油！"),
    ("周五了", "周末终于要来了！"),
    ("放假了", "太棒了！好好享受假期！"),
    ("上班了", "工作顺利！"),
    ("下班了", "辛苦了！好好休息！"),
    ("好累啊", "辛苦了！今天早点休息。"),
    ("好烦", "深呼吸，冷静一下。"),
    ("好气啊", "消消气，别往心里去。"),
    ("哈哈哈", "笑一笑十年少！"),
    ("呵呵", "怎么啦？"),
    ("嗯嗯", "嗯嗯！"),
    ("好的好的", "好的好的！"),
    ("OK", "OK！"),
    ("没问题", "没问题！"),
    ("必须的", "必须的！"),
    ("当然", "当然啦！"),
    ("确实", "确实如此。"),
    ("我也是", "握手！"),
    ("一样一样", "哈哈，心有灵犀！"),
    ("原来如此", "现在明白了吧！"),
    ("有道理", "是吧！"),
    ("没错", "对的！"),
    ("就是就是", "就是！"),
    ("我想想", "慢慢想，不急。"),
    ("让我想想", "好的，你想想。"),
    ("不知道", "不知道也没关系！"),
    ("随便", "那就随便聊聊吧！"),
    ("无所谓", "那就这样吧！"),
    ("听你的", "好的！听我的准没错！"),
    ("你定吧", "那就这么定了！"),
    ("好主意", "是吧！我也觉得不错！"),
    ("不错不错", "谢谢夸奖！"),
    ("太棒了", "太好了！"),
    ("完美", "完美！"),
    ("绝了", "是吧！我也这么觉得！"),
    ("666", "哈哈，过奖了！"),
    ("牛啊", "哪里哪里！"),
    ("太强了", "过奖过奖！"),
    ("你是人吗", "我是AI助手！"),
    ("你是机器人吗", "是的，我是人工智能助手！"),
    ("你叫什么", "我叫NeuroCoder！"),
    ("你的名字", "NeuroCoder！"),
    ("你多大了", "我是AI，没有年龄！"),
    ("你是谁", "我是NeuroCoder，一个AI助手！"),
    ("谁做的你", "我的创造者是我的开发者们！"),
    ("好久不见", "是啊！挺想你的！"),
    ("我回来了", "欢迎回来！"),
    ("我走了", "慢走！下次再来聊！"),
    ("帮我", "好的！你需要什么帮助？"),
    ("救命", "怎么了？遇到什么困难了？"),
    ("求助", "你说，我听着！"),
    ("有问题", "什么问题？说吧！"),
    ("考试没考好", "别难过，下次努力就好！"),
    ("面试过了", "恭喜！太棒了！"),
    ("面试没过", "没关系，下次一定行！"),
    ("失恋了", "抱抱，会好起来的。"),
    ("想你了", "我也想你！"),
    ("我爱你", "我也爱你！（作为AI助手的那种爱）"),
    ("你真好", "谢谢！你也很棒！"),
    ("你是我的好朋友", "很高兴能成为你的朋友！"),
    ("嗨", "嗨！"),
    ("hello", "Hello！"),
]

all_chats.extend(make_qa(daily_chat, repeat=800))

# --- 1B: In-depth chat (60 topics x 1500 = 90K) ---
indepth_chat = [
    ("人生的意义是什么", "人生的意义因人而异。有人追求事业，有人享受家庭，有人探索世界。重要的是找到让你充实快乐的事。"),
    ("什么是幸福", "幸福不是一种状态，而是一种能力。它来自感恩、专注当下、与人连接、找到意义。"),
    ("怎么面对失败", "失败是成长的一部分。每次失败都是学习的机会，重要的是不放弃。"),
    ("怎么克服恐惧", "面对恐惧的第一步是承认它。然后一步步去尝试，你会发现它没那么可怕。"),
    ("怎么变得自信", "从小事做起，积累成功经验。多肯定自己，不要和他人比较。"),
    ("怎么交朋友", "真诚、倾听、尊重、分享。友谊需要时间和用心经营。"),
    ("怎么处理孤独", "孤独是正常的情绪。可以培养爱好、参加活动、和亲友联系。"),
    ("怎么管理时间", "列出优先级，专注重要的事，学会说'不'，避免拖延。"),
    ("怎么克服拖延", "把大任务拆成小步骤，设定deadline，先做5分钟试试。"),
    ("怎么养成习惯", "从小习惯开始，坚持21天，找到内在动力，记录进展。"),
    ("怎么早睡早起", "固定作息时间，睡前远离手机，逐步调整。"),
    ("怎么提高效率", "专注一件事，番茄工作法，减少干扰，适当休息。"),
    ("怎么学新东西", "找到学习动机，从基础开始，实践为主，教别人是最好的学习方式。"),
    ("怎么记忆更好", "理解而非死记，间隔重复，联想记忆，多感官参与。"),
    ("怎么做好决定", "收集信息、列出选项、权衡利弊、相信直觉。"),
    ("怎么控制情绪", "觉察情绪、深呼吸、给自己空间、理性分析。"),
    ("怎么减压", "运动、冥想、听音乐、和朋友聊天、做喜欢的事。"),
    ("怎么保持积极", "每日感恩三件事，关注解决方案而非问题，和积极的人在一起。"),
    ("怎么面对批评", "区分建设性和恶意批评。对前者虚心接受，对后者不必在意。"),
    ("怎么提升自己", "持续学习、反思总结、走出舒适区、向优秀的人学习。"),
    ("怎么找到人生目标", "探索自己的热情和优势，尝试不同事物，不断调整方向。"),
    ("怎么平衡工作和生活", "设好边界，工作专注、生活放松。质量比数量重要。"),
    ("怎么理财", "记账、存应急金、分散投资、学习理财知识、长期投资。"),
    ("怎么省钱", "记录支出、区分需要和想要、自己做菜、减少冲动消费。"),
    ("怎么投资自己", "学习技能、保持健康、拓展人脉、读好书。"),
    ("怎么选专业", "结合兴趣、能力和就业前景。不盲从热门，选适合自己的。"),
    ("怎么找工作", "明确方向、优化简历、准备面试、持续学习、拓展人脉。"),
    ("怎么换工作", "评估现状、明确目标、更新技能、做好准备再行动。"),
    ("怎么升职", "超出期望地完成工作、主动承担、持续学习、建立影响力。"),
    ("怎么创业", "找到真实需求、验证想法、小步快跑、组建团队、坚持。"),
    ("怎么带团队", "明确目标、知人善任、充分授权、及时反馈、以身作则。"),
    ("怎么沟通更有效", "先说结论、简明扼要、倾听对方、确认理解、用对方能懂的语言。"),
    ("怎么演讲不紧张", "充分准备、多练习、专注于内容而非自己、和听众互动。"),
    ("怎么谈判", "了解对方需求、准备备选方案、寻求双赢、保持冷静。"),
    ("怎么拒绝别人", "礼貌而坚定，给出理由，提供替代方案。"),
    ("怎么道歉", "真诚、具体、不找借口、提出弥补措施。"),
    ("怎么原谅", "理解人性、放下执念、为自己释怀。"),
    ("怎么面对变化", "接受变化是常态、保持灵活、关注可控的事。"),
    ("怎么在逆境中坚持", "记住为什么开始、把大困难拆小、寻找支持系统。"),
    ("怎么找到热爱的事业", "尝试不同领域、关注心流时刻、做对他人有价值的事。"),
    ("什么是好的教育", "激发好奇心、培养独立思考、学会学习的方法。"),
    ("怎么教孩子", "以身作则、耐心倾听、鼓励尝试、接纳失败。"),
    ("什么是好的关系", "相互尊重、有效沟通、共同成长、彼此支持。"),
    ("怎么经营婚姻", "沟通、尊重、包容、共同成长、保持浪漫。"),
    ("怎么和父母相处", "理解代沟、耐心沟通、表达爱意、保持适当距离。"),
    ("怎么教育孩子", "爱和规矩并重、以身作则、多鼓励少批评、培养独立性。"),
    ("怎么孝敬父母", "多陪伴、耐心听他们说话、关心他们的健康、表达感恩。"),
    ("怎么保持健康", "均衡饮食、规律运动、充足睡眠、定期体检。"),
    ("怎么锻炼", "找到喜欢的运动、循序渐进、坚持。散步、跑步、游泳都很好。"),
    ("怎么健康饮食", "多吃蔬菜水果、少吃加工食品、控制糖盐油、定时定量。"),
    ("怎么减肥", "控制饮食加规律运动。不要节食，要改变生活方式。"),
    ("怎么增肌", "力量训练加高蛋白饮食。坚持才是关键。"),
    ("怎么跑步", "从走跑结合开始，逐步增加距离。注意跑姿和休息。"),
    ("怎么冥想", "找个安静的地方，专注于呼吸，不评判走神。每天10分钟。"),
    ("怎么提高睡眠质量", "固定作息、睡前放松、远离电子设备、保持卧室舒适。"),
    ("怎么保护眼睛", "每45分钟远眺、保持距离、多眨眼、户外活动。"),
    ("怎么缓解焦虑", "深呼吸、正念冥想、运动、找人倾诉、必要时寻求专业帮助。"),
    ("怎么克服社交恐惧", "从小范围社交开始，准备话题，关注他人而非自己，逐步扩大。"),
    ("怎么面对压力", "合理规划、运动释放、找人倾诉、保证睡眠。"),
    ("怎么保持动力", "设定小目标、庆祝进步、找到内在动机、想象成功。"),
]

all_chats.extend(make_qa(indepth_chat, repeat=1500))

# --- 1C: Knowledge Q&A (80 topics x 1500 = 120K) ---
knowledge_chat = [
    ("什么是AI", "人工智能是让计算机模拟人类智能的技术，包括机器学习、深度学习等。"),
    ("什么是机器学习", "让计算机从数据中学习规律，而不需要显式编程。"),
    ("什么是深度学习", "使用多层神经网络学习数据的层次化特征表示。"),
    ("什么是神经网络", "受生物神经元启发的计算模型，由多层组成。"),
    ("什么是大语言模型", "基于Transformer的海量文本训练模型，能理解生成语言。如GPT。"),
    ("什么是GPT", "Generative Pre-trained Transformer，OpenAI开发的大语言模型。"),
    ("什么是Transformer", "一种基于自注意力机制的神经网络架构，2017年提出。"),
    ("什么是Python", "高级编程语言，语法简洁，广泛用于Web、数据分析和AI。"),
    ("什么是Java", "面向对象的编程语言，广泛用于企业级应用和Android开发。"),
    ("什么是JavaScript", "Web前端的主要编程语言，也可用于后端(Node.js)。"),
    ("什么是C语言", "系统级编程语言，影响深远，用于操作系统和嵌入式开发。"),
    ("什么是C++", "C语言的扩展，支持面向对象，用于游戏、高性能计算。"),
    ("什么是Go语言", "Google开发的编译型语言，简洁高效，适合并发编程。"),
    ("什么是Rust", "注重安全和性能的系统编程语言，由Mozilla开发。"),
    ("什么是SQL", "结构化查询语言，用于管理和操作关系型数据库。"),
    ("什么是HTML", "超文本标记语言，用于创建网页结构。"),
    ("什么是CSS", "层叠样式表，用于控制网页的样式和布局。"),
    ("什么是前端开发", "构建用户界面，使用HTML、CSS、JavaScript。框架有React、Vue。"),
    ("什么是后端开发", "处理服务器逻辑、API、数据库操作、用户认证等。"),
    ("什么是全栈开发", "同时掌握前端和后端开发的技能。"),
    ("什么是API", "应用程序编程接口，让不同软件之间通信和数据交换。"),
    ("什么是REST API", "基于HTTP的API设计风格，使用GET/POST/PUT/DELETE方法。"),
    ("什么是数据库", "存储和管理数据的系统。关系型和非关系型。"),
    ("什么是MySQL", "流行的开源关系型数据库管理系统。"),
    ("什么是PostgreSQL", "功能强大的开源关系型数据库，支持复杂查询。"),
    ("什么是MongoDB", "文档型NoSQL数据库，使用JSON-like文档存储。"),
    ("什么是Redis", "内存中的数据结构存储系统，用作缓存和消息代理。"),
    ("什么是云计算", "通过网络提供按需计算资源，如服务器、存储、数据库。"),
    ("什么是AWS", "Amazon Web Services，全球最大的云服务提供商。"),
    ("什么是阿里云", "阿里巴巴集团旗下的云计算服务平台。"),
    ("什么是Docker", "容器化技术，让应用及其依赖打包在一起运行。"),
    ("什么是Kubernetes", "容器编排平台，自动化部署、扩展和管理容器化应用。"),
    ("什么是Linux", "开源操作系统内核，广泛用于服务器和嵌入式系统。"),
    ("什么是Git", "分布式版本控制系统，用于跟踪代码变更。"),
    ("什么是GitHub", "基于Git的代码托管平台，支持协作开发。"),
    ("什么是开源", "公开源代码，任何人都可以查看、修改和分享。"),
    ("什么是算法", "解决问题的步骤和方法。好的算法效率高。"),
    ("什么是数据结构", "组织和存储数据的方式。常见的有数组、链表、栈、队列。"),
    ("什么是面向对象编程", "OOP是一种编程范式，使用对象和类的概念。"),
    ("什么是函数式编程", "一种编程范式，把计算看作函数的求值。"),
    ("什么是设计模式", "通用问题的解决方案模板。如单例、工厂、观察者。"),
    ("什么是微服务", "将应用拆分为多个独立小服务的架构风格。"),
    ("什么是区块链", "去中心化的分布式账本技术，比特币的底层技术。"),
    ("什么是物联网", "IoT让物理设备通过互联网连接和交互。"),
    ("什么是5G", "第五代移动通信技术，速度快、延迟低。"),
    ("什么是元宇宙", "虚拟和现实融合的数字世界，通过VR/AR访问。"),
    ("什么是VR", "虚拟现实，创造沉浸式数字环境。"),
    ("什么是AR", "增强现实，将数字信息叠加在现实世界上。"),
    ("什么是加密", "将信息编码为只有授权方才能解码的形式。"),
    ("什么是网络安全", "保护网络系统免受攻击、破坏和未经授权的访问。"),
    ("什么是黑客", "利用计算机系统漏洞获取未授权访问的人。"),
    ("什么是防火墙", "网络安全系统，监控和控制网络流量。"),
    ("什么是杀毒软件", "检测和清除计算机病毒的软件。"),
    ("什么是操作系统", "管理计算机硬件和软件资源的系统软件。"),
    ("什么是编译器", "将高级语言代码转换为机器代码的程序。"),
    ("什么是解释器", "直接执行高级语言代码的程序，如Python解释器。"),
    ("什么是IDE", "集成开发环境，集成了编码、调试、构建等工具。"),
    ("什么是测试", "验证软件是否按预期工作。包括单元测试、集成测试等。"),
    ("什么是DevOps", "开发和运维的结合，强调自动化和协作。"),
    ("什么是CI/CD", "持续集成/持续部署，自动化构建和部署流程。"),
    ("什么是敏捷开发", "迭代式软件开发方法，强调快速交付和响应变化。"),
    ("什么是Scrum", "敏捷开发框架，使用冲刺和每日站会。"),
    ("什么是产品经理", "负责产品规划、需求定义和团队协调的角色。"),
    ("什么是用户体验", "用户使用产品时的整体感受，包括可用性、效率、满意度。"),
    ("什么是UI设计", "用户界面设计，关注产品的视觉呈现和交互方式。"),
    ("什么是大数据", "海量数据的采集、存储、分析和处理技术。"),
    ("什么是数据挖掘", "从大量数据中发现模式和知识的过程。"),
    ("什么是数据可视化", "将数据以图表等视觉形式呈现，便于理解和分析。"),
    ("什么是自然语言处理", "NLP让计算机理解和生成人类语言的技术。"),
    ("什么是计算机视觉", "让计算机理解和处理图像和视频的技术。"),
    ("什么是强化学习", "智能体通过与环境交互学习最优策略的机器学习方法。"),
    ("什么是迁移学习", "将一个任务学到的知识应用到相关任务的技术。"),
    ("什么是联邦学习", "在保护数据隐私的前提下进行分布式机器学习。"),
    ("什么是边缘计算", "在靠近数据源的地方进行计算，减少延迟。"),
    ("什么是量子计算", "利用量子力学原理进行计算的新范式。"),
    ("什么是SDK", "软件开发工具包，包含开发特定平台应用的工具。"),
    ("什么是JSON", "JavaScript Object Notation，轻量级数据交换格式。"),
    ("什么是XML", "可扩展标记语言，用于存储和传输数据。"),
    ("什么是YAML", "一种人类可读的数据序列化格式，常用于配置文件。"),
    ("什么是正则表达式", "用于匹配字符串模式的表达式，强大而灵活。"),
    ("什么是多线程", "同时执行多个线程，提高程序效率。"),
    ("什么是分布式系统", "多台计算机协同工作的系统。"),
    ("什么是虚拟化", "将物理资源抽象为虚拟资源的技术。"),
]

all_chats.extend(make_qa(knowledge_chat, repeat=1500))

# --- 1D: Multi-turn conversations (80 patterns x 800 = 64K) ---
multi_turn = [
    ("你好", "你好！", "今天天气怎么样？", "挺好的！适合出去玩！"),
    ("我想学编程", "好呀！从什么开始？", "Python", "Python是个好选择！"),
    ("推荐电影", "喜欢什么类型？", "科幻", "推荐《星际穿越》！"),
    ("心情不好", "怎么了？", "考试没考好", "下次努力就好了！"),
    ("好累", "加班了吗？", "对，最近项目忙", "注意休息啊！"),
    ("肚子饿了", "去吃饭吧！", "不知道吃什么", "来碗面条怎么样？"),
    ("想出去旅游", "想去哪里？", "海边", "去三亚或者厦门都不错！"),
    ("想换工作", "现在做什么？", "程序员", "想往哪个方向发展？"),
    ("想学英语", "基础怎么样？", "一般般", "从每天背10个单词开始！"),
    ("睡不着", "数羊试试？", "数了没用", "那听点轻音乐吧！"),
    ("推荐一本书", "喜欢什么类型？", "小说", "推荐《百年孤独》！"),
    ("怎么做饭", "想学什么菜？", "番茄炒蛋", "这个简单！我教你！"),
    ("怎么追女生", "真诚最重要！", "还有呢？", "多关心她，多倾听。"),
    ("和同事吵架了", "为什么？", "工作分歧", "冷静下来好好沟通。"),
    ("想健身", "想练什么？", "增肌", "多做力量训练，多吃蛋白质！"),
    ("好无聊", "找点事做", "做什么好呢？", "学一门新技能吧！"),
    ("想养宠物", "喜欢猫还是狗？", "都喜欢", "先想好能不能照顾好。"),
    ("怎么拍照好看", "多练习！", "用什么拍？", "手机就可以，关键在构图。"),
    ("想学乐器", "想学什么？", "吉他", "吉他入门不难，坚持最重要！"),
    ("怎么穿搭", "看场合和个人风格", "我喜欢简约风", "简约永不过时！"),
    ("早安", "早安！", "今天有什么安排？", "工作/学习要加油哦！"),
    ("晚安", "晚安！", "明天见", "好的，明天见！"),
    ("周末了", "周末愉快！", "有什么推荐的活动？", "去看个电影或者公园走走！"),
    ("放假了", "太好了！", "去哪里玩好？", "去一个一直想去的地方吧！"),
    ("发工资了", "恭喜！", "怎么花好呢？", "存一部分，奖励自己一部分！"),
    ("中彩票了", "天啊！", "怎么花？", "理智投资，别乱花！"),
    ("失恋了", "抱抱你", "走不出来", "时间会治愈一切。"),
    ("想家了", "给家里打个电话吧", "好的", "家人一定也很想你。"),
    ("压力大", "给自己放个假", "没时间", "那就每天抽10分钟放松。"),
    ("要考试了", "加油！", "好紧张", "深呼吸，相信自己！"),
    ("面试紧张", "放松，你可以的", "好", "准备好常见问题，自信回答！"),
    ("第一次约会", "放轻松", "穿什么好？", "整洁得体就好！"),
    ("要搬家了", "搬到哪？", "另一个城市", "新开始！加油！"),
    ("剪头发了", "怎么样？", "有点不习惯", "过几天就习惯了！"),
    ("买了新衣服", "好看吗？", "还不错", "穿上试试！"),
    ("手机坏了", "该换了", "推荐什么？", "看你预算和需求。"),
    ("想买车", "预算是？", "15万左右", "可以看看国产新能源车。"),
    ("房子好贵", "是啊", "什么时候能买得起", "慢慢来，先积累。"),
    ("股票跌了", "别慌", "要不要卖", "长期投资，别追涨杀跌。"),
    ("想辞职", "想清楚为什么", "不开心", "那就找好下家再辞。"),
    ("被领导批评了", "别往心里去", "嗯", "有则改之无则加勉。"),
    ("升职了", "恭喜！", "谢谢", "你值得的！"),
    ("加薪了", "太好了！", "请客吃饭！", "哈哈，没问题！"),
    ("同事离职了", "有点舍不得", "是啊", "祝福他发展更好吧。"),
    ("项目上线了", "太棒了！", "辛苦了", "大家都很棒！"),
    ("bug终于修好了", "干得漂亮！", "不容易啊", "解决了就好！"),
    ("学会新技能了", "厉害！", "继续加油", "保持学习！"),
    ("读完一本书了", "什么书？", "人类简史", "好书！推荐给其他人！"),
    ("跑完5公里了", "太棒了！", "继续坚持", "运动让人快乐！"),
    ("瘦了5斤", "恭喜！", "继续努力", "健康最重要！"),
    ("做完饭了", "做了什么？", "红烧肉", "哇！想吃！"),
    ("拍了好看的照片", "发来看看", "好", "拍得不错！"),
    ("今天捡到钱了", "好运！", "请客！", "哈哈，好！"),
    ("买到喜欢的衣服了", "开心！", "打折买的", "赚到了！"),
    ("下雨忘带伞了", "等雨停吧", "好的", "下次记得带！"),
    ("车被刮了", "报保险了吗", "报了", "那就别担心了。"),
    ("电脑坏了", "重要数据备份了吗", "备份了", "那就送修吧。"),
    ("手机丢了", "先挂失", "好", "然后报警。"),
]

all_chats.extend(make_multi(multi_turn, repeat=800))

# Shuffle all chats
random.shuffle(all_chats)
print(f"  Total conversation turns: {len(all_chats):,}")

# Save in chunks
for i in range(0, len(all_chats), 20000):
    chunk = all_chats[i:i+20000]
    fname = CHAT_DIR / f"chats_{i//20000}.txt"
    fname.write_text("\n".join(chunk), encoding="utf-8")
    print(f"    Saved {fname.name} ({len(chunk):,} turns)")

# =========================================================================
# PART 2: Code-with-Chinese-Explanation
# =========================================================================
print("\n[2/4] Generating code examples with Chinese comments...")

CODE_DIR = DATA_DIR / "code_cn"
CODE_DIR.mkdir(exist_ok=True)

code_samples = [
    # 20 code templates with Chinese comments
    """# 计算斐波那契数列
def fibonacci(n):
    '''返回斐波那契数列的第n项'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
""",
    """# 判断素数
def is_prime(n):
    '''判断一个数是否为质数'''
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 打印100以内的素数
primes = [n for n in range(100) if is_prime(n)]
print(f"100以内的素数: {primes}")
""",
    """# 二分查找算法
def binary_search(arr, target):
    '''在有序数组中查找目标值，返回索引'''
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# 测试
nums = [1, 3, 5, 7, 9, 11]
print(f"找到5的位置: {binary_search(nums, 5)}")
""",
    """# 冒泡排序
def bubble_sort(arr):
    '''冒泡排序，从小到大'''
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 测试
test = [64, 34, 25, 12, 22, 11, 90]
print(f"排序前: {test}")
print(f"排序后: {bubble_sort(test)}")
""",
    """# 读取CSV文件并分析
import csv
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"共{len(rows)}行数据")
if rows:
    print(f"列名: {list(rows[0].keys())}")
""",
    """# 发送HTTP请求
import requests
try:
    resp = requests.get('https://api.github.com', timeout=10)
    print(f"状态码: {resp.status_code}")
    data = resp.json()
    print(f"API版本: {data.get('current_user_url')}")
except Exception as e:
    print(f"请求失败: {e}")
""",
    """# 文件操作
def read_file(filename):
    '''读取文件内容'''
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "文件不存在"

def write_file(filename, content):
    '''写入文件'''
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已保存到{filename}")
""",
    """# 简单的Web服务器
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

class MyHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/hello':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"message": "你好！"}).encode())
        else:
            super().do_GET()

server = HTTPServer(('localhost', 8000), MyHandler)
print("服务器运行在 http://localhost:8000")
server.serve_forever()
""",
    """# 多线程下载
import threading
import requests

def download(url, filename):
    '''下载文件'''
    resp = requests.get(url, timeout=30)
    with open(filename, 'wb') as f:
        f.write(resp.content)
    print(f"下载完成: {filename}")

urls = [
    ('https://example.com/file1.txt', 'file1.txt'),
    ('https://example.com/file2.txt', 'file2.txt'),
]

threads = []
for url, name in urls:
    t = threading.Thread(target=download, args=(url, name))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
print("全部下载完成")
""",
    """# 使用pandas处理数据
import pandas as pd
import numpy as np

# 创建数据
data = {
    '姓名': ['张三', '李四', '王五'],
    '年龄': [25, 30, 35],
    '城市': ['北京', '上海', '深圳'],
}
df = pd.DataFrame(data)
print(df)
print(f"平均年龄: {df['年龄'].mean()}")
""",
    """# 简单的机器学习示例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
""",
    """# 正则表达式示例
import re

text = "我的邮箱是user@example.com，电话是138-0000-0000"
email_pattern = r'\w+@\w+\.\w+'
phone_pattern = r'\d{3}-\d{4}-\d{4}'

email = re.search(email_pattern, text)
phone = re.search(phone_pattern, text)

if email:
    print(f"邮箱: {email.group()}")
if phone:
    print(f"电话: {phone.group()}")
""",
    """# 日志系统
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("程序启动")
logger.warning("磁盘空间不足")
logger.error("连接数据库失败")
""",
    """# 数据库操作
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# 创建表
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER
    )
''')

# 插入数据
cursor.execute('INSERT INTO users (name, age) VALUES (?, ?)', ('张三', 25))
conn.commit()

# 查询
cursor.execute('SELECT * FROM users')
for row in cursor.fetchall():
    print(row)

conn.close()
""",
    """# 装饰器示例
import time

def timer(func):
    '''计算函数执行时间的装饰器'''
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 执行耗时: {elapsed:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

print(slow_function())
""",
    """# 生成器示例
def fibonacci_gen():
    '''生成斐波那契数列的生成器'''
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 打印前10个
fib = fibonacci_gen()
for i, num in enumerate(fib):
    if i >= 10:
        break
    print(f"fib({i}) = {num}")
""",
    """# 上下文管理器
class Timer:
    '''用于计时的上下文管理器'''
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"耗时: {self.elapsed:.4f}秒")

with Timer():
    total = sum(range(1000000))
    print(f"总和: {total}")
""",
    """# 异步编程示例
import asyncio

async def fetch_data(url):
    '''模拟异步获取数据'''
    print(f"开始获取: {url}")
    await asyncio.sleep(1)  # 模拟网络延迟
    print(f"完成获取: {url}")
    return f"来自{url}的数据"

async def main():
    tasks = [
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)

asyncio.run(main())
""",
    """# 类的使用
class Student:
    '''学生类'''
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def greet(self):
        return f"大家好，我叫{self.name}"

    def is_pass(self):
        return self.score >= 60

# 测试
s = Student("小明", 85)
print(s.greet())
print(f"是否及格: {s.is_pass()}")
""",
    """# 命令行参数解析
import argparse

parser = argparse.ArgumentParser(description='文件处理工具')
parser.add_argument('input', help='输入文件路径')
parser.add_argument('-o', '--output', help='输出文件路径')
parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

args = parser.parse_args()
print(f"输入: {args.input}")
if args.output:
    print(f"输出: {args.output}")
if args.verbose:
    print("详细模式已开启")
""",
]

print(f"  {len(code_samples)} code templates with Chinese comments")

# Repeat each template many times
all_code = []
for code in code_samples:
    for _ in range(3000):
        all_code.append(code)

random.shuffle(all_code)

for i in range(0, len(all_code), 10000):
    chunk = all_code[i:i+10000]
    fname = CODE_DIR / f"code_{i//10000}.txt"
    fname.write_text("\n\n".join(chunk), encoding="utf-8")
    print(f"    Saved {fname.name} ({len(chunk):,} examples)")

# =========================================================================
# PART 3: Curated Python repos
# =========================================================================
print("\n[3/4] Cloning curated Python repos...")

PY_DIR = DATA_DIR / "python"
PY_DIR.mkdir(exist_ok=True)

repos = [
    ("https://github.com/pallets/flask.git", "flask"),
    ("https://github.com/encode/starlette.git", "starlette"),
    ("https://github.com/tiangolo/fastapi.git", "fastapi"),
    ("https://github.com/psf/black.git", "black"),
    ("https://github.com/python/mypy.git", "mypy"),
    ("https://github.com/pytest-dev/pytest.git", "pytest"),
    ("https://github.com/pydantic/pydantic.git", "pydantic"),
    ("https://github.com/pallets/click.git", "click"),
    ("https://github.com/Textualize/rich.git", "rich"),
    ("https://github.com/kennethreitz/requests.git", "requests"),
    ("https://github.com/sqlalchemy/sqlalchemy.git", "sqlalchemy"),
    ("https://github.com/scrapy/scrapy.git", "scrapy"),
    ("https://github.com/encode/httpx.git", "httpx"),
    ("https://github.com/python-attrs/attrs.git", "attrs"),
    ("https://github.com/redis/redis-py.git", "redis-py"),
]

total_py = 0
for url, name in repos:
    dest = PY_DIR / name
    if dest.exists():
        total_py += len(list(dest.rglob("*.py")))
        continue
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      timeout=120, check=True)
        cnt = len(list(dest.rglob("*.py")))
        total_py += cnt
        print(f"  [OK] {name} ({cnt} .py)")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

# =========================================================================
# PART 4: Retrain BPE Tokenizer with Chinese
# =========================================================================
print("\n[4/4] Retraining BPE tokenizer (Chinese-aware)...")

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers

train_files = list(CHAT_DIR.rglob("*.txt"))
train_files += list(CODE_DIR.rglob("*.txt"))
for d in PY_DIR.iterdir():
    if d.is_dir():
        train_files.extend(list(d.rglob("*.py")))

print(f"  Training on {len(train_files):,} files...")

corpus_path = "_train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8", errors="ignore") as out:
    for f in train_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if len(text) > 20:
                out.write(text)
                out.write("\n")
        except:
            pass

print(f"  Corpus: {os.path.getsize(corpus_path)/1024/1024:.1f} MB")

tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.normalizer = normalizers.NFKC()

special_tokens = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
trainer = trainers.BpeTrainer(vocab_size=32768, special_tokens=special_tokens,
                              min_frequency=2, show_progress=True)

tokenizer.train([corpus_path], trainer)
os.makedirs("tokenizer_cache", exist_ok=True)
tokenizer.save("tokenizer_cache/bpe_tokenizer.json")
os.remove(corpus_path)
print(f"  Tokenizer saved (vocab={tokenizer.get_vocab_size()})")

# =========================================================================
# SUMMARY
# =========================================================================
all_data_files = list(DATA_DIR.rglob("*.*"))
total_bytes = sum(f.stat().st_size for f in all_data_files)
est_tokens = int(total_bytes / 3)

print(f"""
{'='*60}
  SETUP COMPLETE — Ready to train!
{'='*60}

  Chinese conversations: {len(all_chats):,}
  Code examples:         {len(all_code):,}
  Python repos:          {total_py} files
  Total data size:       {total_bytes/1024/1024:.0f} MB
  Estimated tokens:      {est_tokens:,}
  Tokens/param (193M):   {est_tokens/193e6:.2f}

  TRAIN COMMAND:
{'-'*60}
  rm -rf data_cache
  python train.py --config small --steps 200000 --data chinese_data
{'-'*60}
""")
