#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from datetime import datetime

from data_processing import preprocess_dem
from flood_model import simulate_flood
from visualization import visualize_results
from utils import setup_logging, read_config


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SRTM DEM 洪水模拟')
    parser.add_argument('-i', '--input', required=True, help='输入DEM文件路径')
    parser.add_argument('-o', '--output', required=True, help='输出目录')
    parser.add_argument('-w', '--water-level', type=float, required=True,
                        help='水位高程（米）')
    parser.add_argument('-c', '--config', default='config.ini',
                        help='配置文件路径')
    parser.add_argument('--fill-depressions', action='store_true',
                        help='填充DEM中的洼地')
    parser.add_argument('--reproject', action='store_true',
                        help='将DEM重投影到指定投影')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='启用详细输出')

    return parser.parse_args()


def main():
    """主程序入口点"""
    # 解析参数
    args = parse_arguments()

    # 设置日志
    log_file = os.path.join(args.output, f"flood_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file, args.verbose)

    # 如果输出目录不存在则创建
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"创建输出目录: {args.output}")

    # 读取配置
    config = read_config(args.config)

    # 预处理DEM
    logging.info("开始DEM预处理...")
    processed_dem_path = preprocess_dem(
        dem_path=args.input,
        output_dir=args.output,
        fill_depressions=args.fill_depressions,
        reproject=args.reproject,
        target_projection=config.get('processing', 'target_projection', fallback='EPSG:3857')
    )

    # 运行洪水模拟
    logging.info(f"正在运行水位为{args.water_level}米的洪水模拟...")
    flood_depth_path = simulate_flood(
        dem_path=processed_dem_path,
        output_dir=args.output,
        water_level=args.water_level
    )

    # 可视化结果
    logging.info("生成可视化...")
    output_files = visualize_results(
        dem_path=processed_dem_path,
        flood_depth_path=flood_depth_path,
        output_dir=args.output,
        water_level=args.water_level,
        base_map=config.get('visualization', 'base_map', fallback=None)
    )

    logging.info(f"洪水模拟成功完成。结果保存到 {args.output}")
    for output_file in output_files:
        logging.info(f"已生成: {output_file}")


if __name__ == "__main__":
    main()