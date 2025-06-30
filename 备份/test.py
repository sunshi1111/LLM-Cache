import contextlib
import random
import os
import time
import hashlib
import sys
import io
import json
import gc
import torch
from torch import nn

from absl import app
from absl import flags
import numpy as np

from gemma import config
from gemma import gemma3_model
from gemma.model import reset_timer, TOTAL_SECOND_LOOP_DURATION

# Define flags
FLAGS = flags.FLAGS

_CKPT = flags.DEFINE_string(
    'ckpt', None, 'Path to the checkpoint file.', required=True
)
_VARIANT = flags.DEFINE_string('variant', '4b', 'Model variant.')
_DEVICE = flags.DEFINE_string('device', 'cpu', 'Device to run the model on.')
_OUTPUT_LEN = flags.DEFINE_integer(
    'output_len', 50, 'Length of the output sequence.'
)
_SEED = flags.DEFINE_integer('seed', 12345, 'Random seed.')
_QUANT = flags.DEFINE_boolean('quant', False, 'Whether to use quantization.')
_INTERACTIVE = flags.DEFINE_boolean('interactive', True, 'Enable interactive conversation mode.')
_DEBUG = flags.DEFINE_boolean('debug', False, '显示调试信息')

# KV缓存相关参数
_CACHE_DIR = flags.DEFINE_string('cache_dir', './gemma3_cache', '保存KV缓存的本地目录')
_NO_CACHE = flags.DEFINE_boolean('no_cache', False, '禁用KV缓存功能')
_CACHE_ID = flags.DEFINE_string('cache_id', None, '指定KV缓存标识符，若不指定则使用随机生成的ID')

# Crail相关参数
_USE_CRAIL = flags.DEFINE_boolean('use_crail', False, '使用Crail存储KV缓存')
_CRAIL_CACHE_DIR = flags.DEFINE_string('crail_cache_dir', '/kvcache', 'Crail中存储KV缓存的目录')
_CRAIL_JAR = flags.DEFINE_string('crail_jar', 
                              '/home/ms-admin/sunshi/crail-example/target/crail-kvcache-client-1.0-SNAPSHOT-jar-with-dependencies.jar',
                              'Crail客户端JAR路径')
_CRAIL_CONF = flags.DEFINE_string('crail_conf', 
                               '/home/ms-admin/sunshi/crail/conf',
                               'Crail配置目录路径')

# 批量测试参数
_BATCH_MODE = flags.DEFINE_boolean('batch_mode', False, '启用批量测试模式')
_SAVE_RESULTS = flags.DEFINE_boolean('save_results', True, '自动保存批量测试结果')
_RESULT_FILE = flags.DEFINE_string('result_file', None, '测试结果保存文件名(不含扩展名)')

# Define valid model variants
_VALID_MODEL_VARIANTS = ['4b', '12b', '27b_v3']

# Define valid devices
_VALID_DEVICES = ['cpu', 'cuda']


# Validator function for the 'variant' flag
def validate_variant(variant):
  if variant not in _VALID_MODEL_VARIANTS:
    raise ValueError(
        f'Invalid variant: {variant}. Valid variants are:'
        f' {_VALID_MODEL_VARIANTS}'
    )
  return True


# Validator function for the 'device' flag
def validate_device(device):
  if device not in _VALID_DEVICES:
    raise ValueError(
        f'Invalid device: {device}. Valid devices are: {_VALID_DEVICES}'
    )
  return True


# Register the validator for the 'variant' flag
flags.register_validator(
    'variant', validate_variant, message='Invalid model variant.'
)

# Register the validator for the 'device' flag
flags.register_validator('device', validate_device, message='Invalid device.')


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)


def format_message(role, content):
  """Format a message for the model."""
  return f"<start_of_turn>{role} {content}<end_of_turn>\n"


def get_prompt_hash(prompt):
  """生成提示文本的哈希值，用于缓存标识"""
  return hashlib.md5(prompt.encode('utf-8', errors='ignore')).hexdigest()[:12]


def batch_test_with_preset_inputs(model, device, output_len=50, use_cache=True, cache_dir=None, 
                    use_crail=False, crail_cache_dir=None, cache_id=None, debug=False):
    """运行预设输入的批量测试对话，保留对话历史"""
    print("\n===== 开始批量测试对话 =====")
    reset_timer()
    
    # 创建缓存目录
    if use_cache and not use_crail and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # 会话标识符
    session_id = cache_id or int(time.time())
    
    # 在这里预设测试输入（最多20个）
    # 直接在这里填写您的测试输入
    test_inputs = [
        # 测试输入1
        "在当今全球化进程不断加速与信息技术飞速发展的双重背景下，人类社会正经历着前所未有的深刻变革，这种变革不仅体现在经济结构的调整与产业形态的更新，还深刻影响着人们的思维方式、生活习惯以及社会组织形式，构成了一幅错综复杂且充满活力的时代画卷。首先，从经济层面来看，数字经济作为新的增长引擎，正以惊人的速度改变着传统的生产关系与商业模式，人工智能、大数据、云计算、区块链等新兴技术的广泛应用，使得企业的生产效率与创新能力得到空前提升，同时也催生了共享经济、平台经济等新型经济形态，为全球经济增长注入了新的动力；与此同时，这一转变过程中也伴随着诸多挑战，如数字鸿沟的扩大、就业结构的变化以及垄断风险的增加等问题，这些问题若不能得到妥善解决，将可能成为阻碍社会公平与可持续发展的重要因素。其次，从社会层面来看，信息技术的普及与应用深刻改变了人们的社交方式与公共参与模式，社交媒体平台的兴起使得信息传播更加快速与广泛，公民可以更方便地表达观点、获取资讯并参与公共事务的讨论，这在某种程度上促进了公民社会的发展与民主进程的推进；然而，信息茧房效应、算法歧视以及网络暴力等现象也日益引起人们的担忧，如何在保障言论自由的同时维护网络空间的秩序与安全，成为各国政府与科技企业共同面临的重要课题。再者，从文化层面来看，全球化进程的深入推进使得不同文化之间的交流与碰撞更加频繁，一方面促进了文化的多元化与创新，另一方面也引发了关于文化认同与文化安全的思考，特别是在互联网时代，西方主导的信息传播格局对非西方国家的文化主权构成了一定挑战，如何在全球化浪潮中保持文化自信并实现文化的创造性转化与创新性发展，成为许多国家文化战略的核心问题。此外，从生态环境层面来看，尽管科技进步为解决环境问题提供了新的可能，如清洁能源技术的发展、环境监测能力的提升等，但过度依赖技术解决方案而忽视生活方式与消费模式的根本转变，可能导致技术乐观主义的陷阱，真正实现人与自然和谐共生的目标，需要在技术创新的同时，更加注重生态文明理念的培育与可持续发展模式的构建。另外，从国际关系层面来看，信息技术的发展使得国家间的联系更加紧密，但同时也带来了网络安全、数据主权等新型安全问题，大国之间围绕技术领导权与标准制定权的竞争日益激烈，如何在维护国家安全与利益的同时促进国际合作与共同治理，成为当前国际关系中的重要议题。在教育领域，信息技术的应用正在推动教育模式的变革，",
        
        # 测试输入2
        "远程教育、在线学习平台的兴起为教育资源的均衡分配提供了新的可能，但如何确保教育的质量与公平，避免技术应用成为新的教育不平等来源，仍需要政策制定者与教育工作者的持续关注与努力。在卫生健康领域，新冠疫情的全球蔓延使得各国更加重视公共卫生体系的建设与国际合作的重要性，信息技术在疫情监测、病毒研究、远程医疗等方面发挥了重要作用，但同时也凸显了全球卫生治理体系的不足，如何构建更加有效的国际卫生合作机制，成为未来全球治理的重要任务。值得注意的是，在科技快速发展的背景下，关于技术伦理与人文关怀的讨论日益引起重视，特别是在人工智能、基因编辑等前沿技术领域，如何平衡技术进步与伦理约束，确保科技发展始终以人为本，服务于人类福祉，成为科学界与社会各界共同思考的问题。综上所述，在这个充满变革与挑战的时代，我们既需要保持开放包容的态度，积极拥抱新技术、新思想带来的机遇，同时也要保持清醒的头脑，审慎应对可能出现的风险与问题，通过加强国际合作、完善治理体系、培育创新能力以及坚持以人为本的发展理念，共同构建一个更加公平、可持续且具有韧性的未来社会。此外，在日益复杂的国际环境中，各国应当秉持多边主义精神，通过对话协商解决争端，共同应对全球性挑战，如气候变化、恐怖主义、传染病防控等，只有实现各国之间的良性互动与合作共赢，才能为人类社会的持续进步提供坚实保障。同时，在国内治理层面，政府、企业、公民社会等多元主体应当形成合力，通过制度创新、技术应用与文化引导等多种手段，共同推动社会治理能力的现代化，为公民创造更加美好的生活环境。最后，对于每一个个体而言，在这个充满不确定性的时代，保持终身学习的习惯、跨文化交流的能力以及批判性思维的素养，将成为适应变革并把握未来的关键要素，唯有不断提升自身的综合素质与创新能力，才能在新的时代浪潮中立于不败之地，实现个人价值与社会价值的统一",
        
        # 测试输入3
        "在深入探讨这一宏大主题的过程中，我们不得不承认全球化与技术发展对社会结构带来的深远影响已经渗透到政治、经济、文化等各个领域，形成了一种新型的社会生态系统，这个系统中的各个要素相互作用、相互影响，构成了一个高度复杂且充满动态变化的整体。从政治层面来看，互联网与社交媒体的兴起使得公共舆论场域发生了根本性变革，传统的自上而下的信息传播模式被更加扁平化、多元化的网络传播形态所取代，这在一定程度上促进了公民政治参与的广度与深度，但同时也为政治极化、虚假信息传播以及社会分裂提供了可能性；各国政府面临着如何在维护国家安全与保障公民自由之间寻找平衡点的艰巨任务，特别是在数据监管、网络安全以及言论管控等方面，需要制定更加精细且符合时代特征的政策框架。在经济发展方面，随着全球价值链的深度重构，传统的产业边界正在被打破，跨国公司通过全球资源配置不断优化生产效率，而新兴市场国家也在努力提升自身在全球产业链中的位置，从简单的加工制造环节向研发设计、品牌营销等高价值环节攀升；与此同时，各国之间围绕技术标准、市场准入、知识产权保护等议题的竞争与博弈日益激烈，贸易保护主义与技术民族主义的倾向在某些地区有所抬头，这对构建开放、包容的全球经济秩序构成了挑战。在社会治理层面，大数据与人工智能技术的应用正在推动智慧城市、智能交通等新型治理模式的发展，通过实时数据收集与分析，政府能够更精准地识别社会需求、预测风险并做出响应，提升公共服务的效率与质量；然而，这也引发了关于算法公平性、技术中立性以及数字隐私的深刻讨论，如何确保技术应用不会强化已有的社会不平等或产生新的排斥与歧视，成为政策制定者必须认真思考的问题。在文化传承与创新方面，数字技术为文化遗产的保护与传播提供了新的手段，通过虚拟现实、增强现实等技术，人们可以以更加沉浸式的方式体验不同的文化形态；同时，网络文学、数字艺术等新型文化表达形式的出现，也拓展了文化创作的边界与可能性，但如何在技术赋能的同时保持文化的真实性与深度，避免文化的碎片化与表面化，仍是文化工作者需要面对的挑战。在教育改革方面，终身学习的理念正在替代传统的阶段性教育模式，面对快速变化的知识体系与职业需求，人们需要不断更新知识结构、提升技能水平，以适应未来工作环境的需要；这要求教育系统更加注重培养学习者的自主学习能力、批判性思维以及跨学科融合能力，而不仅仅是传授特定的知识点或技能操作。在家庭结构与代际关系方面，随着人口流动性增加与城市化进程深入，传统的大家庭模式逐渐被核心家庭或非传统家庭形式所替代，这对老年人照护、儿童教育以及家庭价值观传承等方面产生了深远影响",
        
        # 测试输入4
        "与此同时，数字技术虽然在一定程度上弥合了家庭成员之间的物理距离，但也可能造成家庭内部的数字代沟和情感疏远，如何在技术辅助下重建家庭纽带与代际互动，成为现代社会亟需解决的问题。在健康医疗领域，基因测序、精准医疗、远程诊疗等技术的发展正在改变传统的医疗服务模式，使得疾病预防、诊断与治疗更加精准化、个性化；然而，医疗资源分配不均、医患信任危机以及医疗伦理困境等问题也日益凸显，如何确保先进医疗技术的普惠性与可及性，避免形成新的健康不平等，是医疗卫生体系改革中需要特别关注的方向。在能源与环境治理方面，面对气候变化的全球性挑战，各国正在积极推动能源结构转型与绿色技术创新，如可再生能源、节能技术、碳捕集与封存等领域取得了显著进展；但在实现碳中和目标的过程中，如何平衡经济发展与环境保护、兼顾发达国家与发展中国家的不同诉求，仍然是国际气候治理的难点所在。在安全领域，随着技术的发展，安全威胁的形式也在不断演变，从传统的军事冲突到网络攻击、恐怖主义、生物安全等非传统安全威胁，安全概念的内涵与外延都在不断扩展；这要求各国在强化自身安全能力的同时，更加注重国际安全合作与共同治理，构建更加包容、均衡的全球安全架构。在伦理与价值观层面，科技发展正在挑战人们对生命、意识、自由等基本概念的理解，特别是在人工智能、脑机接口、基因编辑等前沿领域，人们不得不重新思考人与机器的界限、自然与人工的区分以及干预与尊重的平衡；这需要哲学家、科学家、政策制定者以及普通公民共同参与对话，形成更为广泛的社会共识，为科技发展提供伦理指引。在国际关系与全球治理方面，随着全球性挑战的增多与复杂化，传统的国家为中心的治理模式显得力不从心，需要构建更加多元、灵活的全球治理网络，吸纳政府、国际组织、企业、公民社会等多元主体共同参与决策与行动；同时，在维护多边主义与国际规则的基础上，也需要尊重各国的发展道路选择权与文化多样性，避免简单地将某一种政治经济模式强加于人。总之，在这个充满变革与不确定性的时代，人类面临着前所未有的机遇与挑战，如何在保持创新活力的同时确保发展的包容性与可持续性，如何在推动全球化进程中维护各国的合法权益与文化认同，如何在技术赋能社会的同时保障人的尊严与自由，这些都是需要全人类共同思考与行动的重大课题，只有通过开放对话、相互理解与务实合作，才能共同构建一个更加美好的未来世界",
        
        # 测试输入5
        "在进一步审视当代社会的复杂变革时，我们不能忽视城市化进程对人类聚居形态和生活方式产生的深远影响，全球范围内的城市化浪潮正以前所未有的规模和速度改变着人类的栖息环境，特别是在亚洲、非洲等发展中地区，大规模的人口从农村向城市流动，形成了一系列超大型城市和城市群，这种空间重组不仅改变了物理环境，更深刻影响着社会结构、经济模式以及文化景观；城市作为创新、生产和消费的中心，集聚了大量的人才、资本和信息资源，推动了经济增长和技术创新，但与此同时，城市病问题如交通拥堵、住房紧张、环境污染、社区分化等也日益凸显，如何构建更加宜居、包容、有韧性的城市环境，成为城市规划与治理的核心议题。随着数字技术与城市发展的深度融合，智慧城市的理念正在全球范围内得到实践，通过物联网、大数据分析、人工智能等技术手段，城市管理者能够更加精准地掌握城市运行状态，优化资源配置，提升服务质量，但这也要求更高水平的系统集成能力和数据治理框架，以确保技术应用真正服务于城市的可持续发展目标。在农村发展方面，虽然城市化趋势明显，但农村地区仍然是大量人口的生活空间，也是粮食生产、生态保护和文化传承的重要载体，如何通过乡村振兴战略，促进农业现代化、生活便利化和治理精细化，缩小城乡差距，构建城乡融合发展的新格局，是许多国家面临的重要课题。在产业结构调整方面，随着数字经济的崛起和服务业比重的提高，许多传统产业正经历着转型升级的阵痛，一方面，数字化、智能化技术的应用正在重塑制造业的生产方式和商业模式，推动其向高端化、服务化、绿色化方向发展；另一方面，平台经济、共享经济等新业态的兴起，也为就业形态和消费模式带来了革命性变化，但这种变革过程中也伴随着劳动关系的复杂化、市场监管的挑战以及数据垄断的风险，需要更加灵活而有效的政策响应。在科技创新体系建设方面，国家创新能力已经成为国际竞争力的核心要素，各国都在积极构建更加开放、协同、高效的创新生态系统，加强基础研究投入，促进产学研深度融合，培育创新型人才，同时通过知识产权保护、科技金融支持等政策工具，为创新活动提供良好的制度环境；然而，在全球科技竞争日益激烈的背景下，如何平衡开放合作与自主可控，如何确保科技创新的成果能够更加公平地惠及全人类，而不是成为加剧国际分化与对抗的因素，是国际科技治理面临的重要挑战。在金融体系演进方面，随着金融科技的快速发展，传统金融服务的边界正在被打破，移动支付、网络借贷、数字货币等新型金融服务方式极大地提高了金融服务的可及性和便捷性，",
        
        # 测试输入6
        "但也带来了系统性风险的新形态和监管的盲区，特别是在加密货币、去中心化金融等领域，如何在鼓励创新的同时确保金融稳定和消费者权益保护，成为全球金融监管者共同面临的难题。在人口结构变化方面，许多国家正在经历或即将面临人口老龄化的挑战，这不仅影响养老保障体系的可持续性，也对医疗卫生服务、家庭结构、劳动力市场等多个方面产生深远影响；与此同时，人口流动的全球化趋势也在重塑各国的人口格局，移民问题既涉及经济发展、劳动力供给等实际需求，也关联到国家认同、社会整合等深层次议题，如何在尊重多元文化的同时促进社会凝聚力建设，是许多移民目的地国家面临的重要课题。在社会福利制度改革方面，各国都在探索如何在经济全球化和人口结构变化的背景下，构建更加公平、可持续的社会保障体系，既能为弱势群体提供必要的保护和支持，又不会对财政造成过重负担；特别是在劳动力市场日益灵活化、就业形态多样化的背景下，如何设计更加普惠而灵活的社会保险制度，如何平衡政府、市场和社会在福利供给中的角色，成为社会政策创新的重要方向。在身份认同与社会整合方面，随着全球交流的加深和社会流动性的增强，人们的身份认同变得更加多元和复杂，既包括传统的民族、宗教、地域认同，也包括基于职业、阶层、生活方式等因素形成的新型认同；如何在尊重多样性的基础上促进不同群体之间的理解与对话，防止社会因认同差异而走向极化和分裂，是构建和谐社会的重要前提。在公共卫生体系建设方面，新冠疫情的全球大流行不仅暴露了现有全球卫生治理机制的不足，也促使各国重新思考本国公共卫生应急体系的韧性和有效性；未来，如何加强疾病预防控制能力，完善医疗服务网络，建立更加敏捷、协调的危机响应机制，将成为各国卫生体系改革的重点方向。在消费文化与生活方式层面，随着物质生活水平的提高和价值观念的变化，人们的消费行为和生活方式也在发生转变，从注重数量到追求质量，从单纯满足物质需求到更加重视精神和情感体验；同时，可持续消费、健康生活等理念也在逐渐深入人心，影响着市场供给和产品设计的方向。在媒体格局与公共话语空间变迁方面，社交媒体的兴起改变了信息生产和传播的生态，每个人都可以成为内容创造者，参与公共议题的讨论，这在一定程度上促进了言论多样性和公民参与，但同时也带来了信息过载、内容碎片化以及观点极化等问题；如何在保障言论自由的前提下建立更加健康、理性的网络公共空间，成为媒体素养教育和平台治理的重要课题",
        
        # 测试输入7
        "一方面，大国战略竞争的加剧使得国际战略稳定面临新的挑战，核武器、太空武器、高超音速武器等战略武器的发展使得军备竞赛呈现新的态势；另一方面，恐怖主义、极端主义、网络安全、生物安全等非传统安全威胁的全球性扩散也在不断突破国家边界的限制，对全球安全环境构成多重压力。在这样的背景下，仅依靠单一国家或传统的军事同盟难以有效应对复杂多变的安全挑战，需要建立更加包容、多元的安全合作机制，通过对话协商化解分歧，通过合作共赢应对共同威胁。同时，安全与发展的内在联系也日益凸显，只有通过促进包容性发展，消除贫困、不平等等安全风险的根源，才能从根本上实现持久和平与共同安全。在文化交流与文明对话方面，全球化进程既促进了不同文化之间的交流互鉴，也在一定程度上引发了文化同质化和身份焦虑。一方面，数字技术的发展使得文化产品和信息能够以前所未有的速度和规模跨越地理边界，促进了全球文化的融合与创新；另一方面，作为对全球化带来的不确定性和身份危机的反应，文化保守主义和民族主义情绪在一些地区有所抬头，对多元文化共存构成了一定挑战。在这种情况下，如何在尊重文化多样性的基础上，寻求不同文明之间的共通点，开展平等互尊的文明对话，对于构建人类命运共同体具有重要意义。特别是在后疫情时代，面对全球性挑战，各国更需要超越文化和意识形态的分歧，增进相互理解与信任，共同应对人类面临的共同威胁和挑战。在教育变革与人才培养方面，随着知识更新速度的加快和职业需求的变化，传统的以知识传授为中心的教育模式正面临前所未有的挑战。未来教育需要更加注重培养学习者的批判性思维、创造力、合作能力以及终身学习的能力，使其能够在不确定性和变革中保持适应力和竞争力。与此同时，教育的普及化、终身化和个性化趋势也日益明显，从学前教育到老年教育，从正规教育到非正规教育，多样化的教育形式正在满足不同群体、不同阶段的学习需求。数字技术在教育领域的广泛应用，如在线学习平台、教育大数据分析、自适应学习系统等，也为教育模式创新提供了新的可能性，但也对教师角色、教育评价以及教育公平等方面提出了新的课题。在心理健康与精神文明建设方面，现代社会的快节奏、高压力特征以及社会关系的复杂化，使得心理健康问题日益凸显。抑郁症、焦虑症等心理障碍的发病率在全球范围内呈上升趋势，特别在青少年群体中尤为明显。这一现象既与现代生活方式和工作压力相关，也与社会支持系统的弱化和价值观的迷失有关。因此，加强心理健康教育、完善心理咨询服务体系、促进工作生活平衡，成为现代社会治理的重要内容。",
        # 测试输入8
        "总结一下你说的",
        
        # 测试输入9
        "你说的非常不错"
        
        # 测试输入10
        
        
        # 测试输入11
        
        
        # 测试输入12
        
        
        # 测试输入13
        
        
        # 测试输入14
        
        
        # 测试输入15
        
        
        # 测试输入16
        
        
        # 测试输入17
        
        
        # 测试输入18
        
        
        # 测试输入19
        
        
        # 测试输入20
    ]
    
    # 过滤掉空输入
    test_inputs = [input for input in test_inputs if input.strip()]
    
    if not test_inputs:
        print("没有设置任何测试输入，测试结束")
        return
    
    print(f"\n共有 {len(test_inputs)} 个测试输入")
    print("开始进行批量测试...\n")
    
    # 保存测试结果
    results = []
    
    # 对话历史
    conversation_history = ""
    
    # 最近的KV缓存路径和提示
    last_kv_cache_path = None
    last_full_prompt = None
    last_full_prompt_hash = None
    
    # 缓存路径历史记录（保留最近两轮）
    kv_cache_paths = []
    
    # 对每个输入进行测试
    for idx, user_input in enumerate(test_inputs):
        print(f"===== 测试 #{idx+1} =====")
        print(f"用户: {user_input}")
        
        # 构建当前完整对话
        current_full_prompt = conversation_history
        current_full_prompt += format_message("user", user_input)
        current_full_prompt += "<start_of_turn>model"
        
        # 计算当前提示的哈希值
        current_full_prompt_hash = get_prompt_hash(current_full_prompt)
        
        # 构建缓存路径
        if use_crail:
            current_cache_path = f"{crail_cache_dir}/batch_{session_id}_test_{idx+1}_{current_full_prompt_hash}.pt"
        else:
            current_cache_path = os.path.join(cache_dir, f"batch_{session_id}_test_{idx+1}_{current_full_prompt_hash}.pt") if cache_dir else None
        
        if debug:
            print(f"[调试] 测试 #{idx+1} - 提示哈希: {current_full_prompt_hash}")
            if use_cache:
                print(f"[调试] 当前缓存路径: {current_cache_path}")
        
        # 生成参数
        gen_kwargs = {}
        
        # 设置KV缓存加载和保存路径
        if use_cache:
            # 尝试复用上一轮缓存
            if idx > 0 and last_kv_cache_path and last_full_prompt:
                # 找到上一轮提示与当前提示的公共部分长度
                common_prefix_len = 0
                for i in range(min(len(last_full_prompt), len(current_full_prompt))):
                    if last_full_prompt[i] == current_full_prompt[i]:
                        common_prefix_len += 1
                    else:
                        break
                        
                if debug and common_prefix_len > 0:
                    common_prefix = current_full_prompt[:common_prefix_len]
                    print(f"[调试] 发现公共前缀: {common_prefix_len} 字符")
                    print(f"[调试] 公共前缀结束于: \"{common_prefix[-20:] if len(common_prefix) >= 20 else common_prefix}\"")
                
                if common_prefix_len > 0:
                    if use_crail:
                        gen_kwargs["load_kv_cache_crail_path"] = last_kv_cache_path
                        if debug:
                            print(f"[调试] 从Crail加载上一轮KV Cache: {last_kv_cache_path}")
                    else:
                        gen_kwargs["load_kv_cache_path"] = last_kv_cache_path
                        if debug:
                            print(f"[调试] 从本地加载上一轮KV Cache: {last_kv_cache_path}")
            
            # 保存本轮缓存
            if use_crail:
                gen_kwargs["save_kv_cache_crail_path"] = current_cache_path
                if debug:
                    print(f"[调试] 将保存KV Cache到Crail: {current_cache_path}")
            else:
                gen_kwargs["save_kv_cache_path"] = current_cache_path
                if debug:
                    print(f"[调试] 将保存KV Cache到本地: {current_cache_path}")
        
        # 生成回复
        reset_timer()  # 重置计时器
        start_time = time.time()
        result = model.generate(
            [[current_full_prompt]],
            device,
            output_len=output_len,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            **gen_kwargs
        )
        end_time = time.time()
        
        # 提取模型回复
        model_response = result[0].split("<start_of_turn>model")[-1].strip()
        print(f"助手: {model_response}")
        
        # 计算生成时间
        generation_time = end_time - start_time
        print(f"生成时间: {generation_time:.2f} 秒")
        
        
        # 保存结果
        results.append({
            "input": user_input,
            "response": model_response,
            "time": generation_time,
            "second_loop_time": TOTAL_SECOND_LOOP_DURATION
        })
        
        # 更新对话历史
        conversation_history += format_message("user", user_input)
        conversation_history += format_message("model", model_response)
        
        # 更新缓存路径历史
        if use_cache and not use_crail and current_cache_path:
            kv_cache_paths.append(current_cache_path)
            
            # 删除上上轮缓存（保留最近两轮）
            if len(kv_cache_paths) > 2:
                old_cache_path = kv_cache_paths.pop(0)  # 移除并获取最旧的缓存路径
                try:
                    if os.path.exists(old_cache_path):
                        os.remove(old_cache_path)
                        if debug:
                            print(f"[调试] 已删除旧缓存: {old_cache_path}")
                except Exception as e:
                    if debug:
                        print(f"[调试] 删除旧缓存失败: {str(e)}")
        
        # 更新最近的缓存路径和提示
        last_kv_cache_path = current_cache_path
        last_full_prompt = current_full_prompt
        last_full_prompt_hash = current_full_prompt_hash
    
    # 显示测试总结
    print("\n===== 批量测试完成 =====")
    print(f"总测试数: {len(results)}")
    if results:
        avg_time = sum(r["time"] for r in results) / len(results)
        avg_second_loop_time = sum(r["second_loop_time"] for r in results) / len(results)
        
    
    # 自动保存结果
    if _SAVE_RESULTS.value:
        result_filename = _RESULT_FILE.value or f"gemma3_batch_test_{session_id}"
        result_file = f"{result_filename}.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"批量测试结果 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"总测试数: {len(results)}\n")
            if results:
                f.write(f"平均生成时间: {avg_time:.2f} 秒\n")
                f.write(f"平均第二循环时间: {avg_second_loop_time:.6f} 秒\n\n")
            
            for idx, r in enumerate(results):
                f.write(f"===== 测试 #{idx+1} =====\n")
                f.write(f"用户: {r['input']}\n")
                f.write(f"助手: {r['response']}\n")
                f.write(f"生成时间: {r['time']:.2f} 秒\n")
                f.write(f"第二循环时间: {r['second_loop_time']:.6f} 秒\n\n")
        
        # 保存JSON格式结果，便于后续分析
        json_file = f"{result_filename}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_tests": len(results),
                "avg_time": avg_time,
                "avg_second_loop_time": avg_second_loop_time,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"测试结果已保存至: {result_file} 和 {json_file}")


def interactive_chat(model, device, output_len=50, use_cache=True, cache_dir=None, 
                    use_crail=False, crail_cache_dir=None, cache_id=None, debug=False):
  """运行交互式对话，支持KV缓存复用，并自动删除过期缓存"""
  print("\n===== 开始交互式对话 =====")
  print("输入 'exit' 或 'quit' 结束对话")
  reset_timer()
  # 创建缓存目录
  if use_cache and not use_crail and cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
  
  # 会话标识符
  session_id = cache_id or int(time.time())
  
  # 对话历史
  conversation_history = ""
  
  # 缓存路径历史记录（保留最近两轮）
  kv_cache_paths = []
  
  # 最近的KV缓存路径和提示
  last_kv_cache_path = None
  last_full_prompt = None
  last_full_prompt_hash = None
  
  turn = 0
  
  while True:
    turn += 1
    user_input = input("\n用户: ")
    if user_input.lower() in ["exit", "quit"]:
      print("对话结束")
      break
    if turn > 1:  # 第一轮已经在函数开始时重置过了
      reset_timer()
    # 构建当前完整对话
    current_full_prompt = conversation_history
    current_full_prompt += format_message("user", user_input)
    current_full_prompt += "<start_of_turn>model"
    
    # 计算当前提示的哈希值
    current_full_prompt_hash = get_prompt_hash(current_full_prompt)
    
    # 构建缓存路径
    if use_crail:
      current_cache_path = f"{crail_cache_dir}/session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt"
    else:
      current_cache_path = os.path.join(cache_dir, f"session_{session_id}_turn_{turn}_{current_full_prompt_hash}.pt") if cache_dir else None
    
    if debug:
      print(f"\n[调试] 轮次 {turn} - 完整对话提示")
      print(f"[调试] 提示哈希: {current_full_prompt_hash}")
      if use_cache:
        print(f"[调试] 当前缓存路径: {current_cache_path}")
    
    # 生成参数
    gen_kwargs = {}
    
    # 设置KV缓存加载和保存路径
    if use_cache:
      # 尝试复用上一轮缓存
      if turn > 1 and last_kv_cache_path and last_full_prompt:
        # 找到上一轮提示与当前提示的公共部分长度
        common_prefix_len = 0
        for i in range(min(len(last_full_prompt), len(current_full_prompt))):
          if last_full_prompt[i] == current_full_prompt[i]:
            common_prefix_len += 1
          else:
            break
            
        if debug and common_prefix_len > 0:
          common_prefix = current_full_prompt[:common_prefix_len]
          print(f"[调试] 发现公共前缀: {common_prefix_len} 字符")
          print(f"[调试] 公共前缀结束于: \"{common_prefix[-20:] if len(common_prefix) >= 20 else common_prefix}\"")
        
        if common_prefix_len > 0:
          if use_crail:
            gen_kwargs["load_kv_cache_crail_path"] = last_kv_cache_path
            if debug:
              print(f"[调试] 从Crail加载上一轮KV Cache: {last_kv_cache_path}")
          else:
            gen_kwargs["load_kv_cache_path"] = last_kv_cache_path
            if debug:
              print(f"[调试] 从本地加载上一轮KV Cache: {last_kv_cache_path}")
      
      # 保存本轮缓存
      if use_crail:
        gen_kwargs["save_kv_cache_crail_path"] = current_cache_path
        if debug:
          print(f"[调试] 将保存KV Cache到Crail: {current_cache_path}")
      else:
        gen_kwargs["save_kv_cache_path"] = current_cache_path
        if debug:
          print(f"[调试] 将保存KV Cache到本地: {current_cache_path}")
    
    # 生成回复
    start_time = time.time()
    result = model.generate(
      [[current_full_prompt]],
      device,
      output_len=output_len,
      temperature=0.7,
      top_p=0.95,
      top_k=50,
      **gen_kwargs
    )
    end_time = time.time()
    
    # 提取模型回复
    model_response = result[0].split("<start_of_turn>model")[-1].strip()
    print(f"\n助手: {model_response}")
    
    if debug:
      print(f"[调试] 生成时间: {end_time - start_time:.2f} 秒")
    
    # 更新对话历史
    conversation_history += format_message("user", user_input)
    conversation_history += format_message("model", model_response)
    
    # 更新缓存路径历史
    if use_cache and not use_crail and current_cache_path:
      kv_cache_paths.append(current_cache_path)
      
      # 删除上上轮缓存（保留最近两轮）
      if len(kv_cache_paths) > 2:
        old_cache_path = kv_cache_paths.pop(0)  # 移除并获取最旧的缓存路径
        try:
          if os.path.exists(old_cache_path):
            os.remove(old_cache_path)
            if debug:
              print(f"[调试] 已删除旧缓存: {old_cache_path}")
        except Exception as e:
          if debug:
            print(f"[调试] 删除旧缓存失败: {str(e)}")
    
    # 更新最近的缓存路径和提示
    last_kv_cache_path = current_cache_path
    last_full_prompt = current_full_prompt
    last_full_prompt_hash = current_full_prompt_hash


def main(_):
  # Construct the model config.
  model_config = config.get_model_config(_VARIANT.value)
  model_config.dtype = 'float32'
  model_config.quant = _QUANT.value

  # Seed random.
  random.seed(_SEED.value)
  np.random.seed(_SEED.value)
  torch.manual_seed(_SEED.value)

  # 设置KV缓存选项
  use_cache = not _NO_CACHE.value
  use_crail = _USE_CRAIL.value
  cache_dir = _CACHE_DIR.value if use_cache and not use_crail else None
  crail_cache_dir = _CRAIL_CACHE_DIR.value if use_cache and use_crail else None
  cache_id = _CACHE_ID.value
  debug = _DEBUG.value
  
  # 设置Crail环境变量
  if use_crail:
    os.environ["CRAIL_KVCACHE_JAR"] = _CRAIL_JAR.value
    os.environ["CRAIL_CONF_DIR"] = _CRAIL_CONF.value
  
  # 缓存信息提示
  if use_cache:
    if use_crail:
      print(f"Crail KV缓存已启用，存储目录: {crail_cache_dir}")
    else:
      print(f"本地KV缓存已启用，存储目录: {cache_dir}")
      os.makedirs(cache_dir, exist_ok=True)

  # Create the model and load the weights.
  device = torch.device(_DEVICE.value)
  with _set_default_tensor_type(model_config.get_dtype()):
    model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    model.load_weights(_CKPT.value)
    model = model.to(device).eval()
  print('模型加载完成')

  if _INTERACTIVE.value:
    if _BATCH_MODE.value:
      # 运行预设输入的批量测试模式
      batch_test_with_preset_inputs(
        model, 
        device, 
        _OUTPUT_LEN.value, 
        use_cache=use_cache,
        cache_dir=cache_dir,
        use_crail=use_crail,
        crail_cache_dir=crail_cache_dir,
        cache_id=cache_id,
        debug=debug
      )
    else:
      # 启动交互式对话模式
      interactive_chat(
        model, 
        device, 
        _OUTPUT_LEN.value, 
        use_cache=use_cache,
        cache_dir=cache_dir,
        use_crail=use_crail,
        crail_cache_dir=crail_cache_dir,
        cache_id=cache_id,
        debug=debug
      )
  else:
    # 常规测试样例（非交互式）
    # Generate text only.
    result = model.generate(
        [
            [
                '<start_of_turn>user The capital of Italy'
                ' is?<end_of_turn>\n<start_of_turn>model'
            ],
            [
                '<start_of_turn>user What is your'
                ' purpose?<end_of_turn>\n<start_of_turn>model'
            ],
        ],
        device,
        output_len=_OUTPUT_LEN.value,
    )

    # Print the results.
    print('======================================')
    print(f'Text only RESULT: {result}')
    print('======================================')


if __name__ == '__main__':
  app.run(main)