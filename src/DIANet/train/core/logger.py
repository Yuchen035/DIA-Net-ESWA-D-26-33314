# -*- coding: utf-8 -*-
"""
日志模块
"""
import logging
import sys

def get_logger(name: str = "train", level: str = "INFO") -> logging.Logger:
    """
    获取日志器
    
    参数：
        name: 日志器名称
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）
    
    返回：
        logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加 handler
    if not logger.handlers:
        # 控制台输出
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # 格式化
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger
