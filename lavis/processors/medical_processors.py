# import re

# from lavis.common.registry import registry
# from lavis.processors.base_processor import BaseProcessor
# from lavis.processors.randaugment import RandomAugment
# from omegaconf import OmegaConf
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode

# @registry.register_processor("medical_report_processor")
# class MedicalReportProcessor(BaseProcessor):
#    def __init__(self, prompt="", max_words=100):
#        self.prompt = prompt
#        self.max_words = max_words

#    def __call__(self, report):
#        report = self.prompt + self.process_medical_report(report)
#        return report

#    @classmethod 
#    def from_config(cls, cfg=None):
#        if cfg is None:
#            cfg = OmegaConf.create()

#        prompt = cfg.get("prompt", "")
#        max_words = cfg.get("max_words", 100)

#        return cls(prompt=prompt, max_words=max_words)

#    def process_medical_report(cle, report):
        
#         report = report.lower().strip()
        
#         # Handling line breaks and extra spaces
#         report = report.replace('\n', ' ')
#         report = re.sub(r'\s{2,}', ' ', report)
        
#         # Remove number
#         number_patterns = [
#             (r'\d+\.\s*', ''),  # Remove single digit numbers, like 1.
#             (r'\.\s*\d+\.\s*', '. ')  # Removing sentence numbers
#         ]
#         for pattern, replacement in number_patterns:
#             report = re.sub(pattern, replacement, report)
        
#         # Keep only letters, numbers, periods, and commas
#         report = re.sub(r'[^\w\s.,]', '', report) 
        
#         # Standardize punctuation
#         report = re.sub(r'\.+', '.', report)  # Multiple periods become one
#         report = re.sub(r',+', ',', report)   # Multiple commas become one
        
       
#         report = re.sub(r'\s*\.\s*', '. ', report)  # Space after period
#         report = re.sub(r'\s*,\s*', ', ', report)   # Space after comma
        
#         report = report.strip()
#         if not report.endswith('.'):
#             report += '.'
        
#         return report

#    def post_process_report(self, report):
#        """
#        Optional: Post-process generated report for better formatting
#        """
#        # Capitalize first letter of each sentence
#        sentences = report.split('. ')
#        sentences = [s.capitalize() for s in sentences if s]
#        report = '. '.join(sentences)
       
#        # Ensure report ends with period
#        if report and not report.endswith('.'):
#            report += '.'
           
#        return report
   
# from transformers import AutoTokenizer

# # 加载Bio Clinical BERT tokenizer
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# # 测试含标点的医疗文本
# text = "The lungs are clear bilaterally, no pleural effusion. Heart size is normal."

# # 查看分词结果
# tokens = tokenizer.tokenize(text)
# print("Tokens:", tokens)

# def process_medical_report(report):
        
#         report = report.lower().strip()
        
#         # Handling line breaks and extra spaces
#         report = report.replace('\n', ' ')
#         report = re.sub(r'\s{2,}', ' ', report)
        
#         # Remove number
#         number_patterns = [
#             (r'\d+\.\s*', ''),  # Remove single digit numbers, like 1.
#             (r'\.\s*\d+\.\s*', '. ')  # Removing sentence numbers
#         ]
#         for pattern, replacement in number_patterns:
#             report = re.sub(pattern, replacement, report)
        
#         # Keep only letters, numbers, periods, and commas
#         report = re.sub(r'[^\w\s.,]', '', report) 
        
#         # Standardize punctuation
#         report = re.sub(r'\.+', '.', report)  # Multiple periods become one
#         report = re.sub(r',+', ',', report)   # Multiple commas become one
        
       
#         report = re.sub(r'\s*\.\s*', '. ', report)  # Space after period
#         report = re.sub(r'\s*,\s*', ', ', report)   # Space after comma
        
#         report = report.strip()
#         if not report.endswith('.'):
#             report += '.'
        
#         return report

# # 测试用例
# test_reports = [
#    # 1. 包含多种标点和格式问题的报告
#    """FINDINGS:  The heart is enlarged!!!    
#    1. Lungs are clear bilaterally;;
#    2. No pleural effusion/pneumothorax
#    3. Normal mediastinal contours???""",
   
#    # 2. 包含多余空格和换行的报告
#    """The chest x-ray shows:
   
#    heart size is normal,   lungs are clear    bilaterally,
   
#    no evidence of pneumonia...""",
   
#    # 3. 包含特殊字符和编号的报告
#    """1) Heart: mildly enlarged
#    2) Lungs: clear w/o infiltrates
#    3) Pleura: no effusions
#    4) Bones: [unremarkable]""",
   
#    # 4. 包含多余标点和不规范格式的报告
#    """IMPRESSION,,,,: Heart size is normal.... Lungs are clear!!!
#    **No pleural effusion**
#    (Bones appear intact)""",
   
#    # 5. 包含医学缩写和特殊符号的报告
#    """Pt's CXR demonstrates:
#    - Normal cardiac silhouette w/ CTR <50%
#    - B/L clear lungs
#    - No PE or PTX noted"""
# ]

# # 测试函数
# print("测试报告处理函数：\n")
# for i, report in enumerate(test_reports, 1):
#    print(f"原始报告 {i}:\n{report}\n")
#    processed = process_medical_report(report)
#    print(f"处理后报告 {i}:\n{processed}\n")
#    print("-" * 80 + "\n")