import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer


class DataPreprocessor:
    def __init__(self):
        self.tokenizer = 0

    def get_x_y(self, msg_data):
        """
        #     Cleaning data, fitting tokenizer, and extracting X features and y labels
        #
        #     :param data: pd.DataFrame
        #                 Raw messages data with spam or ham labels
        #     :return: pd.DataFrame
        #                 X features and y labels
        #     """
        msg_data = self.clean_messages(msg_data)

        self.__fit_tokenizer(msg_data)

        data_prepared = self.get_tokenized(msg_data)

        X = data_prepared
        y = msg_data['v1']

        return X, y

    def clean_messages(self, msg_data):
        """
            Cleaning each message from digits, removing multiple spaces

            :param data: pd.DataFrame data messages
            :return: pd.DataFrame cleaned messages
            """

        def clean_message(msg):
            msg = msg.lower().strip()
            msg = re.sub(r'[\W]', ' ', msg)
            msg = re.sub(r'[\d]', '', msg)
            msg = re.sub(r'\s+', ' ', msg)
            return msg

        msg_data['v2'] = msg_data['v2'].apply(clean_message)

        return msg_data

    def __fit_tokenizer(self, msg_data):
        """
        Fitting model tokenizer with general dat words

        :param msg_data: pd.DataFrame
                        cleaned data ready for fitting tokenizer
        """
        number_of_unique_words = pd.Series(np.concatenate(msg_data['v2'].apply(str.split).to_numpy())).nunique()
        self.tokenizer = CountVectorizer(max_features=number_of_unique_words)
        self.tokenizer.fit(msg_data['v2'])

    def get_tokenized(self, msg_data):
        """
        Tokin

        :param msg_data: pd.DataFrame
                        with cleaned messages data
        :return: pd.DataFrame
                Tokenized messages
        """

        if self.tokenizer == 0:
            raise Exception('Not implemented exception, call gat_x_y and fit the data to tokenizer')

        tokenized_words = self.tokenizer.transform(msg_data['v2']).toarray()
        tokenized_words_df = pd.DataFrame(columns=self.tokenizer.get_feature_names(), data=tokenized_words)

        return tokenized_words_df

