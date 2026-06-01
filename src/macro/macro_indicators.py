import os
import logging
import pandas as pd
import akshare as ak
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MacroSentinel:
    """
    Sentinel 模块：负责宏观经济与流动性因子的监控与数据清洗
    """
    def __init__(self, data_dir: str = "C:/Users/Administrator/fundamental_quant_framework/data/macro"):
        self.data_dir = data_dir
        # 确保数据存储目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_china_10y_yield(self, start_date: str = "20150101") -> pd.DataFrame:
        """
        从中债数据源抓取10年期国债到期收益率历史数据
        :param start_date: 格式 'YYYYMMDD', 起始抓取日期
        :return: 清洗后的 DataFrame [date, yield_10y]
        """
        logging.info(f"开始抓取10年期国债收益率数据，起点日期: {start_date}...")
        try:
            # 使用 AkShare 获取中债国债收益率曲线历史数据
            # 该接口返回包含不同期限的国债到期收益率
            df_raw = ak.bond_zh_us_rate(start_date=start_date)[["日期", "中国国债收益率10年"]]
            
            if df_raw.empty:
                raise ValueError("抓取到的数据为空")

            # 2. 数据清洗与过滤
            # 目标字段：'日期' 和 '10年' 期限的收益率
            target_cols = {'日期': 'date', '中国国债收益率10年': 'zh_yield_10y'}
            df_cleaned = df_raw[list(target_cols.keys())].rename(columns=target_cols)
            
            # 类型转换
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
            df_cleaned['zh_yield_10y'] = pd.to_numeric(df_cleaned['zh_yield_10y'], errors='coerce')
            
            # 去除空值并按日期正序排列
            df_cleaned = df_cleaned.dropna().sort_values('date').reset_index(drop=True)
            
            # 3. 基础宏观特征工程（计算衍生指标）
            # 计算 20 日和 60 日滚动均线，用以判断宏观利率趋势（拐点监测）
            df_cleaned['zh_yield_10y_ma20'] = df_cleaned['zh_yield_10y'].rolling(window=20).mean()
            df_cleaned['zh_yield_10y_ma60'] = df_cleaned['zh_yield_10y'].rolling(window=60).mean()
            
            # 计算当前利率在历史（如滚动1年/250交易日）中的百分分位数，评估利率水位线
            df_cleaned['zh_yield_10y_percentile_250d'] = (
                df_cleaned['zh_yield_10y']
                .rolling(window=250)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 250 else None)
            )

            logging.info(f"数据抓取与衍生指标计算完成，共 {len(df_cleaned)} 条记录。")
            return df_cleaned

        except Exception as e:
            logging.error(f"抓取10年期国债收益率失败: {str(e)}")
            return pd.DataFrame()

    def fetch_us_10y_yield(self, start_date: str = "20150101") -> pd.DataFrame:
        """
        从中债数据源抓取10年期国债到期收益率历史数据
        :param start_date: 格式 'YYYYMMDD', 起始抓取日期
        :return: 清洗后的 DataFrame [date, yield_10y]
        """
        logging.info(f"开始抓取10年期国债收益率数据，起点日期: {start_date}...")
        try:
            # 使用 AkShare 获取中债国债收益率曲线历史数据
            # 该接口返回包含不同期限的国债到期收益率
            df_raw = ak.bond_zh_us_rate(start_date=start_date)[["日期", "美国国债收益率10年"]]
            
            if df_raw.empty:
                raise ValueError("抓取到的数据为空")

            # 2. 数据清洗与过滤
            # 目标字段：'日期' 和 '10年' 期限的收益率
            target_cols = {'日期': 'date', '美国国债收益率10年': 'us_yield_10y'}
            df_cleaned = df_raw[list(target_cols.keys())].rename(columns=target_cols)
            
            # 类型转换
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
            df_cleaned['us_yield_10y'] = pd.to_numeric(df_cleaned['us_yield_10y'], errors='coerce')
            
            # 去除空值并按日期正序排列
            df_cleaned = df_cleaned.dropna().sort_values('date').reset_index(drop=True)
            
            # 3. 基础宏观特征工程（计算衍生指标）
            # 计算 20 日和 60 日滚动均线，用以判断宏观利率趋势（拐点监测）
            df_cleaned['us_yield_10y_ma20'] = df_cleaned['us_yield_10y'].rolling(window=20).mean()
            df_cleaned['us_yield_10y_ma60'] = df_cleaned['us_yield_10y'].rolling(window=60).mean()
            
            # 计算当前利率在历史（如滚动1年/250交易日）中的百分分位数，评估利率水位线
            df_cleaned['us_yield_10y_percentile_250d'] = (
                df_cleaned['us_yield_10y']
                .rolling(window=250)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 250 else None)
            )

            logging.info(f"数据抓取与衍生指标计算完成，共 {len(df_cleaned)} 条记录。")
            return df_cleaned

        except Exception as e:
            logging.error(f"抓取10年期国债收益率失败: {str(e)}")
            return pd.DataFrame()
        

    def fetch_china_money_supply_indicators(self, start_date: str = "20150101") -> pd.DataFrame:
        """
        从东方财富抓取中国货币供应量原始数据，并计算核心金融观测指标（M2-M1剪刀差、货币活期化比率等）
        :param start_date: 格式 'YYYYMMDD', 起始抓取日期
        :return: 包含衍生指标的增强型 DataFrame
        """
        logging.info(f"开始从东方财富获取货币供应量并计算核心衍生指标，起点日期: {start_date}...")
        try:
            # 1. 抓取原始数据
            df_raw = ak.macro_china_money_supply()
            
            if df_raw.empty:
                raise ValueError("东方财富货币供应量接口返回数据为空")
                
            # 2. 动态字段锚定 (模糊匹配，防止接口列名微调导致崩溃)
            try:
                date_col = [c for c in df_raw.columns if '月份' in str(c) or '月' in str(c)][0]
                m2_val_col = [c for c in df_raw.columns if 'M2' in str(c) and '数量' in str(c)][0]
                m2_yoy_col = [c for c in df_raw.columns if 'M2' in str(c) and '同比' in str(c)][0]
                m1_val_col = [c for c in df_raw.columns if 'M1' in str(c) and '数量' in str(c)][0]
                m1_yoy_col = [c for c in df_raw.columns if 'M1' in str(c) and '同比' in str(c)][0]
            except IndexError as ie:
                raise KeyError(f"接口字段匹配失败，请检查原始列名是否变更。原始列名: {df_raw.columns.tolist()}") from ie

            # 3. 提取核心列并重命名
            rename_dict = {
                date_col: 'date',
                m2_val_col: 'm2_value',
                m2_yoy_col: 'm2_yoy',
                m1_val_col: 'm1_value',
                m1_yoy_col: 'm1_yoy'
            }
            df_cleaned = df_raw[list(rename_dict.keys())].rename(columns=rename_dict)
            
            # 4. 数据类型严格转换与清洗
            # 针对东方财富可能返回的 "2026年04月份" 或 "2026.04" 进行鲁棒性清洗
            # 正则表达式含义：把“年”、“月份”、“月”或者“.”都统一替换为横杠“-”
            df_cleaned['date'] = df_cleaned['date'].astype(str)\
                .str.replace(r'[年月份.]', '-', regex=True)\
                .str.replace(r'-+', '-', regex=True)\
                .str.rstrip('-')
                
            # 补齐日线级别（比如 2026-04 补齐为 2026-04-01）
            df_cleaned['date'] = df_cleaned['date'] + '-01'
            
            # 显式指定 format 格式，彻底消除 UserWarning 警告并大幅提升解析速度
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%Y-%m-%d', errors='coerce')

            # 5. 【高价值金融特征工程 - 终极改良版】
            # 指标一：M2 - M1 剪刀差 (不变)
            df_cleaned['m2_m1_scissors'] = df_cleaned['m2_yoy'] - df_cleaned['m1_yoy']
            
            # 指标二：M1 / M2 货币活性比率 (不变)
            df_cleaned['m1_m2_ratio'] = df_cleaned['m1_value'] / df_cleaned['m2_value']
            
            # 指标三：M2 绝对增量同比变动 (基于当年增速反推，彻底解决倒序和单位断层问题)
            # 东方财富 2025-2026 最新数据单位已调整，我们动态计算：
            # 1. 先根据 m2_yoy 反推出去年同期的 M2 理论值
            m2_last_year = df_cleaned['m2_value'] / (1 + df_cleaned['m2_yoy'] / 100)
            # 2. 计算差值，并根据当前数据量级自动适配单位（转化为万亿元）
            # 如果 m2_value 超过 300 万，说明单位是 10亿元，除以 1000 转化成万亿元
            # 如果 m2_value 在 30 万左右，说明单位是 亿元，除以 10000 转化成万亿元
            df_cleaned['m2_net_expansion_trillion'] = df_cleaned.apply(
                lambda row: (row['m2_value'] - (row['m2_value'] / (1 + row['m2_yoy'] / 100))) / 1000 
                if row['m2_value'] > 1000000 
                else (row['m2_value'] - (row['m2_value'] / (1 + row['m2_yoy'] / 100))) / 10000, 
                axis=1
            )

            # 6. 按照指定时间切片输出
            start_dt = pd.to_datetime(start_date)
            df_final = df_cleaned[df_cleaned['date'] >= start_dt].reset_index(drop=True)
            
            # 优化输出格式，保留 4 位有效小数
            float_cols = ['m2_yoy', 'm1_yoy', 'm2_m1_scissors', 'm1_m2_ratio', 'm2_net_expansion_trillion']
            df_final[float_cols] = df_final[float_cols].round(4)
            
            logging.info(f"货币供应量高级指标计算完成，共生成 {len(df_final)} 条宏观时序特征。")
            return df_final

        except Exception as e:
            logging.error(f"货币供应量核心指标特征工程构建失败: {str(e)}")
            return pd.DataFrame()
        
    def fetch_china_pmi(self, start_date: str = "20150101") -> pd.DataFrame:
        """
        从金十数据源抓取中国官方制造 PMI 历史数据并清洗
        :param start_date: 格式 'YYYYMMDD', 起始抓取日期
        :return: 清洗后的 DataFrame [date, pmi]
        """
        logging.info(f"开始抓取中国官方制造业 PMI 数据，起点日期: {start_date}...")
        try:
            # 1. 抓取原始数据
            df_raw = ak.macro_china_pmi_yearly()
            
            if df_raw.empty:
                raise ValueError("抓取到的 PMI 数据为空")
            
            # 2. 数据清洗与格式化 (该接口 index 为时间字符串，列名为 'value')
            df_raw = df_raw.reset_index()
            target_cols = {'日期': 'date', '今值': 'pmi'}
            df_cleaned = df_raw[['日期', '今值']].rename(columns=target_cols)
            
            # 类型转换
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
            df_cleaned['pmi'] = pd.to_numeric(df_cleaned['pmi'], errors='coerce')
            
            # 过滤时间段并正序排列
            start_dt = pd.to_datetime(start_date)
            df_cleaned = df_cleaned[df_cleaned['date'] >= start_dt]
            df_cleaned = df_cleaned.dropna().sort_values('date').reset_index(drop=True)
            
            # 3. 量化特征工程
            # 计算 PMI 的环比动量 (本月变动了多少个点)，用以捕捉二阶拐点
            df_cleaned['pmi_mom_change'] = df_cleaned['pmi'].diff(1)
            
            # 标记荣枯线状态：1 代表扩张(>50)，0 代表收缩(<=50)
            df_cleaned['pmi_expansion_flag'] = df_cleaned['pmi'].apply(lambda x: 1 if x > 50.0 else 0)
            
            # 四位小数规范化
            df_cleaned[['pmi', 'pmi_mom_change']] = df_cleaned[['pmi', 'pmi_mom_change']].round(4)
            
            logging.info(f"中国官方 PMI 数据抓取与动量清洗完成，共 {len(df_cleaned)} 条记录。")
            return df_cleaned

        except Exception as e:
            logging.error(f"抓取中国官方 PMI 失败: {str(e)}")
            return pd.DataFrame()

    def fetch_china_cpi(self, start_date: str = "20150101") -> pd.DataFrame:
        """
        从东方财富抓取中国居民消费价格指数（CPI）月度数据，并计算核心通胀观测指标
        :param start_date: 格式 'YYYYMMDD', 起始抓取日期
        :return: 包含 CPI 同比、环比及衍生动量指标的增强型 DataFrame
        """
        logging.info(f"开始抓取中国 CPI 月度数据，起点日期: {start_date}...")
        try:
            # 1. 抓取原始数据 —— AkShare CPI 月度接口
            # 返回字段通常包含：日期、全国-当月、全国-同比增长、全国-环比增长、核心CPI等
            df_raw = ak.macro_china_cpi_monthly()

            if df_raw.empty:
                raise ValueError("东方财富 CPI 接口返回数据为空")

            # 2. 动态字段锚定 (模糊匹配，防止接口列名微调导致崩溃)
            try:
                date_col = [c for c in df_raw.columns if '日期' in str(c) or '月份' in str(c)][0]
                # 全国 CPI 当月值（定基指数）
                cpi_current_col = [c for c in df_raw.columns if '全国' in str(c) and '当月' in str(c)][0]
                # 全国 CPI 同比
                cpi_yoy_col = [c for c in df_raw.columns if '全国' in str(c) and '同比' in str(c)][0]
                # 全国 CPI 环比
                cpi_mom_col = [c for c in df_raw.columns if '全国' in str(c) and '环比' in str(c)][0]
            except IndexError as ie:
                raise KeyError(f"CPI 接口字段匹配失败，请检查原始列名是否变更。原始列名: {df_raw.columns.tolist()}") from ie

            # 3. 提取核心列并重命名
            rename_dict = {
                date_col: 'date',
                cpi_current_col: 'cpi_index',
                cpi_yoy_col: 'cpi_yoy',
                cpi_mom_col: 'cpi_mom'
            }
            df_cleaned = df_raw[list(rename_dict.keys())].rename(columns=rename_dict)

            # 4. 数据类型严格转换与清洗
            # 日期列鲁棒解析 —— 兼容 "2026年05月份"、"2026.05" 等格式
            df_cleaned['date'] = df_cleaned['date'].astype(str)\
                .str.replace(r'[年月份.]', '-', regex=True)\
                .str.replace(r'-+', '-', regex=True)\
                .str.rstrip('-')
            # 补齐日为当月首日
            df_cleaned['date'] = df_cleaned['date'] + '-01'
            df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], format='%Y-%m-%d', errors='coerce')

            # 数值列转换
            for col in ['cpi_index', 'cpi_yoy', 'cpi_mom']:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

            # 5. 高价值通胀特征工程
            # 指标一：CPI 同比的 3 个月 / 6 个月滚动均值（判断通胀趋势方向）
            df_cleaned['cpi_yoy_ma3'] = df_cleaned['cpi_yoy'].rolling(window=3).mean()
            df_cleaned['cpi_yoy_ma6'] = df_cleaned['cpi_yoy'].rolling(window=6).mean()

            # 指标二：CPI 环比动量 —— 连续正向/负向月数（捕捉趋势持续性）
            def consecutive_sign_streak(series: pd.Series) -> pd.Series:
                """计算环比连续同向的累计强度（正值累加，负值累减，0 重置）"""
                streak = []
                acc = 0
                for val in series:
                    if pd.isna(val):
                        streak.append(None)
                        continue
                    if acc == 0 or val * acc > 0:
                        acc += val
                    elif val == 0:
                        acc = 0
                    else:
                        acc = val  # 方向反转，重新计数
                    streak.append(round(acc, 4))
                return pd.Series(streak, index=series.index)

            df_cleaned['cpi_mom_streak'] = consecutive_sign_streak(df_cleaned['cpi_mom'])

            # 指标三：CPI 同比加速度（二阶导数）—— 通胀放缓/加速的拐点信号
            df_cleaned['cpi_yoy_accel'] = df_cleaned['cpi_yoy'].diff(1)

            # 指标四：CPI 同比在滚动 3 年（36个月）中的历史分位数 —— 评估当前通胀水位
            df_cleaned['cpi_yoy_percentile_36m'] = (
                df_cleaned['cpi_yoy']
                .rolling(window=36)
                .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 36 else None)
            )

            # 指标五：核心通胀警戒标记 —— 当 CPI 同比突破 3% 或跌破 0%（通缩）时触发
            def inflation_alert(yoy: float) -> int:
                if pd.isna(yoy):
                    return 0
                if yoy >= 3.0:
                    return 1   # 高通胀警戒
                elif yoy < 0:
                    return -1  # 通缩警戒
                return 0        # 正常区间

            df_cleaned['cpi_alert'] = df_cleaned['cpi_yoy'].apply(inflation_alert)

            # 6. 按起始日期切片并规范化精度
            start_dt = pd.to_datetime(start_date)
            df_final = df_cleaned[df_cleaned['date'] >= start_dt].reset_index(drop=True)

            float_cols = ['cpi_index', 'cpi_yoy', 'cpi_mom', 'cpi_yoy_ma3', 'cpi_yoy_ma6',
                          'cpi_mom_streak', 'cpi_yoy_accel', 'cpi_yoy_percentile_36m']
            df_final[float_cols] = df_final[float_cols].round(4)

            logging.info(f"CPI 通胀核心指标计算完成，共生成 {len(df_final)} 条月度宏观时序特征。")
            return df_final

        except Exception as e:
            logging.error(f"抓取中国 CPI 数据失败: {str(e)}")
            return pd.DataFrame()

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        full_path = os.path.join(self.data_dir, filename)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        logging.info(f"📂 成功将宏观指标落库至: {full_path}")

# --- 模块独立测试逻辑 ---
if __name__ == "__main__":
    print("Starting macro_indicators.py - MacroSentinel run")
    sentinel = MacroSentinel()

    # 示例调用：保留注释的其他抓取示例以便按需启用

    # 居民消费价格指数CPI（示例，开启网络时会抓取并保存）
    try:
        cpi_df_china = sentinel.fetch_china_cpi(start_date="20100101")
        if cpi_df_china is None:
            print("cpi_df_china is None - fetch_china_cpi returned None")
        elif cpi_df_china.empty:
            print("No CPI data returned (empty DataFrame). Check network or AkShare availability.")
        else:
            print("\n📊 刚刚抓取并生成的宏观快照 (最新5条):")
            print(cpi_df_china.tail())
            sentinel.save_to_csv(cpi_df_china, filename="cpi_df_china.csv")
    except Exception as e:
        import traceback
        print("Exception when fetching CPI:", str(e))
        traceback.print_exc()

    print("MacroSentinel run finished.")