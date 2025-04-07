
import polars as pl
from loglead.loaders.raw import RawLoader
from loglead.enhancers import EventLogEnhancer
from loglead.anomaly_detection import AnomalyDetector

class FileContextSelection:
    def __init__(self, filetype: str, anomaly_detection_method: str, file_path: str, **kwargs):
        self.filetype = filetype
        self.anomaly_detection_method = anomaly_detection_method
        self.file_path = file_path
    
    def load_LO2_file(self) -> pl.DataFrame:
        loader = RawLoader(
            filename=self.file_path,
            timestamp_pattern=r"^(\d{1,2}:\d{2}:\d{2}\.\d{3})",
            timestamp_format="%H:%M:%S.%f",
            missing_timestamp_action="merge"
        )
        loader.load()

        df = loader.df.with_columns([
            pl.col("m_message")
            .str.extract(loader.timestamp_pattern, group_index=1)
            .str.pad_end(12, "0")
            .str.strptime(pl.Datetime, "%H:%M:%S.%f", strict=False)
            .alias("m_timestamp"),

            pl.col("m_message")
            .str.replace(loader.timestamp_pattern, "")
            .alias("m_message"),
        ])

        return df
    
    def enhance(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("m_message").str.extract(r"\[([^\]]+)\]", group_index=1).alias("thread"),
            pl.col("m_message").str.extract(r"\] +(\S+)", group_index=1).alias("request_id"),
            pl.col("m_message").str.extract(r"\] +\S+ +(\w+)", group_index=1).alias("level"),
            pl.col("m_message").str.extract(r"\w+ +(\S+ +<init>)", group_index=1).alias("class_method"),
            pl.col("m_message").str.extract(r"<init> - (.*)", group_index=1).alias("log_text")
        ])

        df = df.fill_null("")

        enhancer = EventLogEnhancer(df)
        df = enhancer.normalize()
        df = enhancer.parse_spell()
        df = enhancer.words()
        df = enhancer.trigrams()
        #df = enhancer.parse_drain()
        #df = enhancer.parse_tip()
        #df = enhancer.parse_pliplom()
        #df = enhancer.parse_iplom()
        #df = enhancer.parse_brain()
        df = enhancer.length()

        return df
    
    def get_LOF_anomalies(self, df: pl.DataFrame):
        anomaly_detector = AnomalyDetector(
            item_list_col="e_words",
            numeric_cols=["e_words_len", "e_trigrams_len", "e_chars_len", "e_lines_len", "e_event_id_len"],
            store_scores=True,
            print_scores=False,
            auc_roc=False
        )

        anomaly_detector.test_train_split(df, test_frac=0.90, shuffle=True)
        anomaly_detector.train_LOF()
        df = anomaly_detector.predict()

        return df

    def get_context(self):
        if self.filetype == "LO2":
            df = self.load_LO2_file()
        else:
            return None

        df = self.enhance(df)

        if self.anomaly_detection_method == "LOF":
            df = self.get_LOF_anomalies(df)
            df = df.rename({"pred_ano": "context"})
            return df
        else:
            return None
