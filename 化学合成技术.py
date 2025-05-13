# 定义简单的元素及其化合价
elements = {
    "H": {"valences": [#citation-1](citation-1)},
    "O": {"valences": [-2]},
    "Na": {"valences": [#citation-1](citation-1)},
    "Cl": {"valences": [-1]},
    "C": {"valences": [4, -4]},
    "N": {"valences": [-3, 3, 5]},
}
# 化合物字典，用于存储已知的简单化合物
compounds = {
    "H2O": {"H": 2, "O": 1},
    "NaCl": {"Na": 1, "Cl": 1},
    "CH4": {"C": 1, "H": 4},
    "NH3": {"N": 1, "H": 3},
}

# 根据化合价推断化学反应的方法
def suggest_synthesis(target_compound):
    if target_compound not in compounds:
        print(f"抱歉，化合物 {target_compound} 未知。")
        return
  
    target_elements = compounds[target_compound]
    reaction_steps = []

    # 推导反应步骤
    for element, required_count in target_elements.items():
        if element not in elements:
            continue
        
        valence_options = elements[element] ["valences"]
        
        # 查找匹配的化合价使得总和为零
        for valence in valence_options:
            if required_count * valence == 0:
                continue

            multiple = abs(required_count // valence)
            if required_count + multiple * valence == 0:
                reaction_steps.append((multiple, element, valence))
                break

    # 生成假设的化学方程式
    if reaction_steps:
        reaction_ingredients = " + ".join([f"{step[0]} {step[#citation-1](citation-1)}" for step in reaction_steps])
        print(f"为了生成 {target_compound}，可能的合成反应为：")
        print(f"{reaction_ingredients} -> {target_compound}")
    else:
        print(f"没有找到可行的方法来合成 {target_compound}")

# 示例化合物合成
for compound in compounds:
    suggest_synthesis(compound)
    print("-" * 40)
