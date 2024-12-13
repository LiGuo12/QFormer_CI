import re

def convert_dependency_rules(input_file, output_file):
    """
    Convert dependency rules with fixed regex patterns
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
        
        print("Original content sample:")
        print(content[:200])
        
        # 1. 处理特殊情况
        special_replacements = {
            r'compound': 'nn',
            r'acl:relcl': 'rcmod',
            r'conj:and\.or': 'conj_and.or'
        }
        
        for old, new in special_replacements.items():
            # 匹配 {dependency:/pattern/} 格式
            pattern = r'\{dependency:/' + old + r'/\}'
            content_before = content
            content = re.sub(pattern, '{dependency:/' + new + '/}', content)
            if content != content_before:
                print(f"\nReplaced {old} with {new}")
        
        # 2. 处理通用模式
        print("\nBefore nmod replacement:")
        print(re.findall(r'\{dependency:/nmod:[^}]+/\}', content))
        
        # 替换 nmod:* 为 prep_*
        content = re.sub(r'\{dependency:/nmod:([^}]+)/\}', r'{dependency:/prep_\1/}', content)
        
        # 替换 conj:* 为 conj_*（排除已处理的特殊情况）
        content = re.sub(r'\{dependency:/conj:(?!and\.or)([^}]+)/\}', r'{dependency:/conj_\1/}', content)
        
        print("\nAfter replacements, checking for prep_ patterns:")
        print(re.findall(r'prep_[^}/]+', content[:1000]))
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"\nWritten to {output_file}")
        return True
        
    except Exception as e:
        print(f"转换过程中出现错误: {str(e)}")
        return False

input_file = r'D:\code\ubc\paper\new_paper\code\LAVIS_llm_contrastive\chexpert_labeler\patterns\pre_negation_uncertainty.txt'
output_file = r'D:\code\ubc\paper\new_paper\code\LAVIS_llm_contrastive\chexpert_labeler\patterns\new_pre_negation_uncertainty.txt'

convert_dependency_rules(input_file, output_file)