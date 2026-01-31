"""
========================================
文本编码器模块
========================================

使用预训练BERT模型提取标题的语义特征
"""

import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    """
    文本编码器
    
    使用预训练的BERT模型提取标题的语义特征
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        
        # 冻结BERT的嵌入层和前几个Transformer层
        # 只微调后几层，减少过拟合
        for name, param in self.bert.named_parameters():
            # 只训练最后2层encoder
            if 'encoder.layer.10' not in name and 'encoder.layer.11' not in name:
                param.requires_grad = False
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(config.TEXT_FEATURE_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: BERT输入ID [batch_size, MAX_TEXT_LENGTH]
            attention_mask: 注意力掩码 [batch_size, MAX_TEXT_LENGTH]
            
        Returns:
            文本特征 [batch_size, HIDDEN_DIM]
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]标记的输出作为整个句子的表示
        # [batch_size, 768]
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # 投影到隐藏空间
        # [batch_size, HIDDEN_DIM]
        output = self.projection(cls_output)
        
        return output
