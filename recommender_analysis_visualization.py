import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil
from sklearn.tree import DecisionTreeClassifier


# Cell: Import libraries and set up environment
"""
# Recommender Systems Analysis and Visualization
This notebook performs an exploratory analysis of recommender systems using the Sim4Rec library.
We'll generate synthetic data, compare multiple baseline recommenders, and visualize their performance.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# Set log level to warnings only
spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Cell: Define custom recommender template
"""
## MyRecommender Template
Below is a template class for implementing a custom recommender system.
Students should extend this class with their own recommendation algorithm.
"""

class MyRecommender:
    """
    Template class for implementing a custom recommender.
    
    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """
    
    def __init__(self, seed=None):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        # Add your initialization logic here
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        # Implement your training logic here
        # For example:
        #  1. Extract relevant features from user_features and item_features
        #  2. Learn user preferences from the log
        #  3. Build item similarity matrices or latent factor models
        #  4. Store learned parameters for later prediction
        pass
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        # Implement your recommendation logic here
        # For example:
        #  1. Extract relevant features for prediction
        #  2. Calculate relevance scores for each user-item pair
        #  3. Rank items by relevance and select top-k
        #  4. Return a dataframe with columns: user_idx, item_idx, relevance
        
        # Example of a random recommender implementation:
        # Cross join users and items
        recs = users.crossJoin(items)
        
        # Filter out already seen items if needed
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx")
            recs = recs.join(
                seen_items,
                on=["user_idx", "item_idx"],
                how="left_anti"
            )
        
        # Add random relevance scores
        recs = recs.withColumn(
            "relevance",
            sf.rand(seed=self.seed)
        )
        
        # Rank items by relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs


from sklearn.tree import DecisionTreeClassifier

class DecisionTreeRecommender:
    def __init__(self, seed=42, max_depth=5, min_samples_leaf=1, min_samples_split=2, ccp_alpha=0.0):
        self.seed = seed
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.model = None
        self.user_features = None
        self.item_features = None
        self.feature_names = None

    def fit(self, log, user_features=None, item_features=None):
        self.user_features = user_features
        self.item_features = item_features

        df = (
            log.join(user_features, on="user_idx")
               .join(item_features, on="item_idx")
               .toPandas()
        )

        # Drop duplicated columns, just in case
        df = df.loc[:, ~df.columns.duplicated()]

        # Extract features and labels
        X = df.drop(columns=["user_idx", "item_idx", "relevance"])
        X = pd.get_dummies(X)
        X = X.loc[:, ~X.columns.duplicated()]
        y = df["relevance"].astype(int)

        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            ccp_alpha=self.ccp_alpha,
            random_state=self.seed
        )
        self.model.fit(X, y)
        self.feature_names = self.model.feature_names_in_

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        user_feats = user_features or self.user_features
        item_feats = item_features or self.item_features

        recs = users.crossJoin(items)
        recs = recs.join(user_feats, on="user_idx").join(item_feats, on="item_idx")

        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx")
            recs = recs.join(seen, on=["user_idx", "item_idx"], how="left_anti")


        recs_pd = recs.toPandas()
        recs_pd = recs_pd.loc[:, ~recs_pd.columns.duplicated()]


        if "price" not in recs_pd.columns:
            price_cols = [col for col in recs_pd.columns if "price" in col.lower()]
            if not price_cols:
                raise ValueError("No price column found in prediction dataframe.")
            recs_pd["price"] = recs_pd[price_cols[0]]

        recs_pd["price"] = pd.to_numeric(recs_pd["price"], errors="coerce").fillna(0.0)

        X_pred = recs_pd.drop(columns=["user_idx", "item_idx", "price"])
        X_pred = pd.get_dummies(X_pred)
        X_pred = X_pred.loc[:, ~X_pred.columns.duplicated()]
        X_pred = X_pred.reindex(columns=self.feature_names, fill_value=0)

        recs_pd["relevance"] = self.model.predict_proba(X_pred)[:, 1]
        recs_pd["revenue_score"] = recs_pd["relevance"] * recs_pd["price"]
        recs_pd["rank"] = recs_pd.groupby("user_idx")["revenue_score"].rank(ascending=False, method="first")

        topk = recs_pd[recs_pd["rank"] <= k][["user_idx", "item_idx", "relevance"]]

        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        return spark.createDataFrame(topk)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sim4rec.utils import pandas_to_spark
class LogisticRegressionRecommender:
    def __init__(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        self.model = LogisticRegression(
            random_state=self.seed, 
            max_iter=1500, 
            solver='saga',
            C=1
        )
        self.scaler = StandardScaler()
        self.user_feats = None
        self.item_feats = None

    def fit(self, log:DataFrame, user_features=None, item_features=None):
        # Create training dataframe
        if user_features and item_features:
            df = log.join(
                user_features, 
                on='user_idx'
            ).join(
                item_features, 
                on='item_idx'
            ).drop(
                'user_idx', 'item_idx', '__iter'
            ).toPandas()
            
            # Scale price
            df = pd.get_dummies(df, dtype=float)
            df['price'] = self.scaler.fit_transform(df[['price']])
            
            # Split X and y
            X = df.drop(['relevance'], axis=1)
            y = df['relevance']
            
            # Fit model to data
            self.model.fit(X,y)
            
            # Store features
            self.user_feats = user_features
            self.item_feats = item_features
            
    def predict(self, log, k, users:DataFrame, items:DataFrame, user_features=None, item_features=None, filter_seen_items=True):
        
        recs = users.join(items).drop('__iter')
        
        # Filter seen items
        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx")
            recs = recs.join(seen, on=["user_idx", "item_idx"], how="left_anti")
            
        # Format data
        recs = recs.toPandas().copy()
        recs = pd.get_dummies(recs, dtype=float)
        recs['orig_price'] = recs['price']
        recs['price'] = self.scaler.transform(recs[['price']])

        # Use model to predict likelihoods
        recs['prob'] = self.model.predict_proba(recs.drop(['user_idx', 'item_idx', 'orig_price'], axis=1))[:,np.where(self.model.classes_ == 1)[0][0]]
        
        # Relevance = purchase probability x price
        recs['relevance'] = recs['prob']*recs['orig_price']
        
        # Filter top k most relevant items
        recs = recs.sort_values(by=['user_idx', 'relevance'], ascending=[True, False])
        recs = recs.groupby('user_idx').head(k)

        recs['price'] = recs['orig_price']
        
        # Convert back to Spark and fix schema types to match original log
        from pyspark.sql.types import LongType
        result = pandas_to_spark(recs)
        result = result.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        result = result.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
       
        return result

class KNNRecommender:
    """
    User-based KNN collaborative filtering recommender.
    Recommends items liked by similar users based on historical interactions.
    """
    def __init__(self, k=5, seed=None):
        self.k = k
        self.seed = seed
        self.user_item_matrix = None
        self.user_sim_matrix = None
        self.user_ids = None
        self.item_ids = None

    def fit(self, log, user_features=None, item_features=None):
        # Convert log to Pandas for similarity computation
        pd_log = log.select("user_idx", "item_idx", "relevance").toPandas()
        if pd_log.empty:
            self.user_item_matrix = None
            self.user_sim_matrix = None
            self.user_ids = None
            self.item_ids = None
            return
        user_item = pd.pivot_table(pd_log, index="user_idx", columns="item_idx", values="relevance", fill_value=0)
        self.user_item_matrix = user_item
        self.user_ids = user_item.index.tolist()
        self.item_ids = user_item.columns.tolist()
        # Compute cosine similarity between users
        from sklearn.metrics.pairwise import cosine_similarity
        self.user_sim_matrix = pd.DataFrame(
            cosine_similarity(user_item),
            index=user_item.index,
            columns=user_item.index
        )

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        import pandas as pd
        from sim4rec.utils import pandas_to_spark
        from pyspark.sql.types import LongType
        import numpy as np
        import pyspark.sql.functions as sf

        # Fallback to PopularityRecommender if not fitted
        if self.user_sim_matrix is None or self.user_item_matrix is None:
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, filter_seen_items=filter_seen_items)

        user_list = users.select("user_idx").toPandas()["user_idx"].tolist()
        item_list = items.select("item_idx").toPandas()["item_idx"].tolist()
        recs = []
        for user_id in user_list:
            if user_id not in self.user_sim_matrix.index:
                # Fallback for users with no history
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), items, user_features, item_features, filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recs.extend(pop_recs_pd.to_dict("records"))
                continue
            # Find k most similar users (excluding self)
            neighbors = (
                self.user_sim_matrix.loc[user_id]
                .drop(user_id, errors='ignore')
                .sort_values(ascending=False)
                .head(self.k)
                .index
            )
            # Aggregate their item preferences
            neighbor_items = self.user_item_matrix.loc[neighbors].sum(axis=0)
            # Remove items already seen by the user
            if user_id in self.user_item_matrix.index:
                seen_items = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
            else:
                seen_items = set()
            candidate_items = neighbor_items.drop(seen_items, errors='ignore')
            # Only recommend items that exist in the current items list
            candidate_items = candidate_items[candidate_items.index.isin(item_list)]
            # Get top-k items
            top_items = candidate_items.sort_values(ascending=False).head(k)
            if top_items.empty:
                # Fallback to popularity if no candidate items
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), items, user_features, item_features, filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recs.extend(pop_recs_pd.to_dict("records"))
            else:
                for item_id, score in top_items.items():
                    recs.append({"user_idx": user_id, "item_idx": item_id, "relevance": float(score)})
        # Convert to DataFrame and then to Spark DataFrame
        recs_df = pd.DataFrame(recs)
        if recs_df.empty:
            # Fallback: recommend popular items if no recs at all
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, filter_seen_items=filter_seen_items)
        # Ensure correct types
        recs_df["user_idx"] = recs_df["user_idx"].astype(np.int64)
        recs_df["item_idx"] = recs_df["item_idx"].astype(np.int64)
        result = pandas_to_spark(recs_df)
        result = result.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        result = result.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
        return result

import torch
import torch.nn as nn
import torch.nn.functional as F
from sim4rec.utils import pandas_to_spark
from pyspark.sql.types import LongType
import pyspark.sql.functions as sf

class TransformerRecommender:
    def __init__(self, embedding_dim=64, n_heads=4, n_layers=2, max_seq_len=50, dropout=0.05, seed=42):
        torch.manual_seed(seed)
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False

    def _init_model(self, n_items):
        self.n_items = n_items
        self.item_embedding = nn.Embedding(n_items + 1, self.embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, n_items + 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)
        self.initialized = True

    def parameters(self):
        return list(self.item_embedding.parameters()) + \
               list(self.pos_embedding.parameters()) + \
               list(self.transformer.parameters()) + \
               list(self.output_layer.parameters())

    def to(self, device):
        self.item_embedding = self.item_embedding.to(device)
        self.pos_embedding = self.pos_embedding.to(device)
        self.transformer = self.transformer.to(device)
        self.norm = self.norm.to(device)
        self.output_layer = self.output_layer.to(device)

    def fit(self, log, user_features=None, item_features=None, epochs=10, batch_size=128):
        df = log.select("user_idx", "item_idx").toPandas()

        if not self.initialized:
            n_items = df["item_idx"].max() + 1
            self._init_model(n_items)

        df["interaction_order"] = df.groupby("user_idx").cumcount()
        df = df.sort_values(["user_idx", "interaction_order"])
        user_sequences = df.groupby("user_idx")["item_idx"].apply(list).values
        user_sequences = [seq for seq in user_sequences if len(seq) >= 2]
        if not user_sequences:
            return

        padded_seqs = [([0] * (self.max_seq_len - len(seq)) + seq[-self.max_seq_len:]) for seq in user_sequences]
        inputs = torch.tensor(padded_seqs, dtype=torch.long).to(self.device)
        targets = inputs.clone()

        price_lookup = None
        if item_features is not None:
            price_lookup = item_features.select("item_idx", "price").toPandas().set_index("item_idx")["price"].to_dict()

        price_tensor = torch.zeros(self.n_items + 1, dtype=torch.float32)
        if price_lookup is not None:
            for item_id, price in price_lookup.items():
                if item_id <= self.n_items:
                    price_tensor[item_id] = price
            price_tensor = price_tensor.to(self.device)

        for epoch in range(epochs):
            for i in range(0, len(inputs), batch_size):
                x = inputs[i:i + batch_size]
                y = targets[i:i + batch_size]
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                seq_len = x.shape[1]
                pos_ids = torch.arange(seq_len).unsqueeze(0).repeat(x.size(0), 1).to(self.device)

                emb = self.item_embedding(x) + self.pos_embedding(pos_ids)
                attention_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
                output = self.transformer(emb, mask=attention_mask)

                pooled = self.norm(output[:, -1, :])
                logits = self.output_layer(pooled)

                probs = torch.softmax(logits, dim=1)
                expected_revenue = (probs * price_tensor.unsqueeze(0)).sum(dim=1)
                loss = -expected_revenue.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        user_seq_df = log.select("user_idx", "item_idx").toPandas()
        user_seq_df["interaction_order"] = user_seq_df.groupby("user_idx").cumcount()
        user_seq_df = user_seq_df.sort_values(["user_idx", "interaction_order"])
        user_seq_dict = user_seq_df.groupby("user_idx")["item_idx"].apply(list).to_dict()

        all_items = items.select("item_idx", "price").toPandas()
        results = []

        for user_id in users.select("user_idx").toPandas()["user_idx"]:
            seq = user_seq_dict.get(user_id, [])
            seq = [0] * (self.max_seq_len - len(seq)) + seq[-self.max_seq_len:]
            input_tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
            pos_ids = torch.arange(self.max_seq_len).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.item_embedding(input_tensor) + self.pos_embedding(pos_ids)
                output = self.transformer(emb)
                pooled = self.norm(output[:, -1, :])
                logits = self.output_layer(pooled).squeeze(0).cpu()

            probs = torch.softmax(logits, dim=0).numpy()
            item_scores = pd.DataFrame({
                "item_idx": range(len(probs)),
                "prob": probs
            }).merge(all_items, on="item_idx", how="inner")

            if filter_seen_items:
                seen_items = user_seq_dict.get(user_id, [])
                item_scores = item_scores[~item_scores["item_idx"].isin(seen_items)]

            item_scores["relevance"] = item_scores["prob"] * item_scores["price"]
            topk = item_scores.sort_values("relevance", ascending=False).head(k)
            topk["user_idx"] = user_id
            results.append(topk[["user_idx", "item_idx", "relevance"]])

        if not results:
            return pandas_to_spark(pd.DataFrame(columns=["user_idx", "item_idx", "relevance"]))

        final_df = pd.concat(results)
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        return spark.createDataFrame(final_df.astype({"user_idx": int, "item_idx": int}))

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

class AutoRegressiveRecommender:
    """
    Auto-Regressive (AR) Sequence-Based Recommender for Checkpoint 2
    
    Implements n-gram autoregressive models that treat the last n items as context
    to predict the next item. Supports multiple orders (1, 2, 3) with additive
    smoothing for regularization.
    
    Key Features:
    - N-gram modeling with configurable order (1=unigram, 2=bigram, 3=trigram)
    - Additive smoothing (add-k) for handling unseen sequences
    - Revenue-aware ranking using expected_revenue = price × probability
    - Sequence construction with configurable max length
    - Robust fallback mechanisms for cold-start scenarios
    """
    
    def __init__(self, order=2, smoothing_alpha=1.0, max_seq_len=50, seed=None):
        """
        Initialize the Auto-Regressive recommender.
        
        Args:
            order (int): N-gram order (1, 2, or 3)
            smoothing_alpha (float): Additive smoothing parameter (add-k smoothing)
            max_seq_len (int): Maximum sequence length to maintain
            seed (int): Random seed for reproducibility
        """
        self.order = max(1, min(3, order))  # Clamp to 1-3 as specified
        self.smoothing_alpha = smoothing_alpha
        self.max_seq_len = max_seq_len
        self.seed = seed
        
        # Data structures for AR modeling
        self.user_sequences = {}  # {user_id: [item_ids]}
        self.item_prices = {}     # {item_id: price}
        self.ngram_counts = {}    # {context_tuple: {next_item: count}}
        self.context_counts = {}  # {context_tuple: total_count}
        self.item_vocab = set()   # All unique items seen
        self.vocab_size = 0
        
    def _extract_sequences(self, log):
        """Extract ordered sequences of user interactions from log data."""
        import pandas as pd
        
        available_cols = log.columns
        
        # Handle different log formats
        if "price" in available_cols and "response" in available_cols:
            df = log.select("user_idx", "item_idx", "price", "response", "__iter").toPandas()
        elif "relevance" in available_cols:
            df = log.select("user_idx", "item_idx", "relevance").toPandas()
            df["response"] = df["relevance"]
            df["price"] = 10.0
            df["__iter"] = "start"
        else:
            df = log.select("user_idx", "item_idx").toPandas()
            df["response"] = 1
            df["price"] = 10.0
            df["__iter"] = "start"
        
        # Sort by user and iteration to maintain sequence order
        df = df.sort_values(["user_idx", "__iter"])
        
        user_sequences = {}
        item_prices = {}
        
        for user_id, group in df.groupby("user_idx"):
            # Only include items that were actually interacted with (response > 0)
            sequence = []
            for _, row in group.iterrows():
                if row['response'] > 0:  # Only positive interactions
                    sequence.append(int(row['item_idx']))
                    item_prices[int(row['item_idx'])] = row['price']
            
            # Keep only recent interactions up to max_seq_len
            if sequence:
                user_sequences[user_id] = sequence[-self.max_seq_len:]
        
        return user_sequences, item_prices
    
    def _build_ngram_model(self):
        """
        Build n-gram transition model from user sequences.
        
        For each sequence, extract n-grams and count transitions:
        - Order 1 (unigram): P(item) 
        - Order 2 (bigram): P(item | prev_item)
        - Order 3 (trigram): P(item | prev_item1, prev_item2)
        """
        self.ngram_counts = {}
        self.context_counts = {}
        self.item_vocab = set()
        
        # Collect all items for vocabulary
        for sequence in self.user_sequences.values():
            self.item_vocab.update(sequence)
        
        self.vocab_size = len(self.item_vocab)
        
        if self.vocab_size == 0:
            return
        
        # Build n-gram counts from all user sequences
        for sequence in self.user_sequences.values():
            if len(sequence) < self.order:
                continue
                
            # Extract n-grams from sequence
            for i in range(len(sequence) - self.order + 1):
                # Context is the first (order-1) items, target is the last item
                if self.order == 1:
                    context = ()  # Empty context for unigram
                    target = sequence[i]
                else:
                    context = tuple(sequence[i:i + self.order - 1])
                    target = sequence[i + self.order - 1]
                
                # Update counts
                if context not in self.ngram_counts:
                    self.ngram_counts[context] = {}
                    self.context_counts[context] = 0
                
                if target not in self.ngram_counts[context]:
                    self.ngram_counts[context][target] = 0
                
                self.ngram_counts[context][target] += 1
                self.context_counts[context] += 1
    
    def _get_context_from_sequence(self, sequence):
        """Extract the appropriate context from a user sequence based on order."""
        if not sequence:
            return () if self.order == 1 else None
        
        if self.order == 1:
            return ()  # Unigram has no context
        elif self.order == 2:
            return (sequence[-1],) if len(sequence) >= 1 else None
        elif self.order == 3:
            return tuple(sequence[-2:]) if len(sequence) >= 2 else None
        
        return None
    
    def _predict_next_item_probabilities(self, context):
        """
        Predict probabilities for next items given context using additive smoothing.
        
        Args:
            context: Context tuple for n-gram lookup
            
        Returns:
            dict: {item_id: probability} for all items in vocabulary
        """
        if context is None or self.vocab_size == 0:
            # Uniform distribution if no context or vocabulary
            uniform_prob = 1.0 / max(1, self.vocab_size)
            return {item: uniform_prob for item in self.item_vocab}
        
        probabilities = {}
        
        # Get counts for this context
        context_total = self.context_counts.get(context, 0)
        context_items = self.ngram_counts.get(context, {})
        
        # Apply additive smoothing: P(item|context) = (count + α) / (total + α * V)
        # where V is vocabulary size and α is smoothing parameter
        denominator = context_total + self.smoothing_alpha * self.vocab_size
        
        for item in self.item_vocab:
            count = context_items.get(item, 0)
            probabilities[item] = (count + self.smoothing_alpha) / denominator
        
        return probabilities
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Fit the Auto-Regressive model.
        
        Extracts user sequences and builds n-gram transition model with
        additive smoothing for regularization.
        """
        if log is None or log.count() == 0:
            self.user_sequences = {}
            self.item_prices = {}
            self.ngram_counts = {}
            self.context_counts = {}
            self.item_vocab = set()
            self.vocab_size = 0
            return
        
        # Extract sequences from log
        self.user_sequences, self.item_prices = self._extract_sequences(log)
        
        # Build n-gram model
        self._build_ngram_model()
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate sequence-aware recommendations using AR model.
        
        For each user:
        1. Extract context from their interaction sequence
        2. Use n-gram model to predict next item probabilities
        3. Rank by expected revenue (probability × price)
        4. Return top-k recommendations
        """
        import pandas as pd
        import numpy as np
        from sim4rec.utils import pandas_to_spark
        from pyspark.sql.types import LongType
        import pyspark.sql.functions as sf

        # Fallback to PopularityRecommender if no model built
        if not self.ngram_counts or self.vocab_size == 0:
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, 
                                 filter_seen_items=filter_seen_items)

        user_list = users.select("user_idx").toPandas()["user_idx"].tolist()
        item_list = items.select("item_idx", "price").toPandas()
        recommendations = []
        
        for user_id in user_list:
            user_sequence = self.user_sequences.get(user_id, [])
            
            # Get context for this user
            context = self._get_context_from_sequence(user_sequence)
            
            if context is None:
                # No valid context: fallback to popularity
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), 
                                         items, user_features, item_features, 
                                         filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recommendations.extend(pop_recs_pd.to_dict("records"))
                continue
            
            # Get next item probabilities from AR model
            item_probabilities = self._predict_next_item_probabilities(context)
            
            # Filter items that exist in the current item catalog
            candidate_items = []
            seen_items = set(user_sequence) if filter_seen_items else set()
            
            for _, item_row in item_list.iterrows():
                item_id = int(item_row['item_idx'])
                price = item_row['price']
                
                # Skip if item not in vocabulary or already seen
                if item_id not in self.item_vocab or item_id in seen_items:
                    continue
                
                # Get probability from AR model
                probability = item_probabilities.get(item_id, 0.0)
                
                # Calculate expected revenue = probability × price
                expected_revenue = probability * price
                
                candidate_items.append({
                    "user_idx": user_id,
                    "item_idx": item_id,
                    "relevance": expected_revenue
                })
            
            # Rank by expected revenue and select top-k
            candidate_items.sort(key=lambda x: x["relevance"], reverse=True)
            top_recommendations = candidate_items[:k]
            
            if not top_recommendations:
                # No candidates: fallback to popularity
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), 
                                         items, user_features, item_features, 
                                         filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recommendations.extend(pop_recs_pd.to_dict("records"))
            else:
                recommendations.extend(top_recommendations)
        
        # Convert to Spark DataFrame with proper types
        if not recommendations:
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, 
                                 filter_seen_items=filter_seen_items)
        
        recs_df = pd.DataFrame(recommendations)
        recs_df["user_idx"] = recs_df["user_idx"].astype(np.int64)
        recs_df["item_idx"] = recs_df["item_idx"].astype(np.int64)
        
        result = pandas_to_spark(recs_df)
        result = result.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        result = result.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
        
        return result
    
class Node2VecRecommender:
    def __init__(self, walk_length=8, num_walks=40, embedding_dim=32, window_size=4, epochs=3, seed=42):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, log, user_features=None, item_features=None):
        import networkx as nx
        from gensim.models import Word2Vec
        import numpy as np

        df = log.select("user_idx", "item_idx").toPandas()
        offset = df["user_idx"].max() + 1
        df["item_idx"] += offset

        self.offset = offset
        self.user_ids = df["user_idx"].unique()
        self.item_ids = df["item_idx"].unique() - offset

        G = nx.Graph()
        G.add_edges_from(df[["user_idx", "item_idx"]].values)

        np.random.seed(self.seed)
        walks = []
        for _ in range(self.num_walks):
            for node in np.random.permutation(G.nodes()):
                walk = [node]
                while len(walk) < self.walk_length:
                    neighbors = list(G.neighbors(walk[-1]))
                    if neighbors:
                        walk.append(np.random.choice(neighbors))
                    else:
                        break
                walks.append([str(n) for n in walk])

        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            sg=1,
            workers=1,
            epochs=self.epochs,
            seed=self.seed
        )

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        import numpy as np
        import pandas as pd
        from sklearn.metrics.pairwise import cosine_similarity
        from sim4rec.utils import pandas_to_spark

        items_df = items.select("item_idx", "price").toPandas()
        item_vecs = np.stack([
            self.model.wv[str(i + self.offset)] if str(i + self.offset) in self.model.wv else np.zeros(self.embedding_dim)
            for i in items_df["item_idx"]
        ])

        results = []
        seen_df = log.select("user_idx", "item_idx").toPandas() if filter_seen_items else pd.DataFrame()

        for user_id in users.select("user_idx").toPandas()["user_idx"]:
            user_key = str(user_id)
            if user_key not in self.model.wv:
                continue
            user_vec = self.model.wv[user_key].reshape(1, -1)
            sims = cosine_similarity(user_vec, item_vecs).flatten()

            item_scores = items_df.copy()
            item_scores["relevance"] = sims * item_scores["price"]

            if filter_seen_items:
                seen_items = seen_df[seen_df["user_idx"] == user_id]["item_idx"].values
                item_scores = item_scores[~item_scores["item_idx"].isin(seen_items)]

            topk = item_scores.nlargest(k, "relevance")[["item_idx", "relevance"]]
            topk["user_idx"] = user_id
            results.append(topk[["user_idx", "item_idx", "relevance"]])

        if results:
            final_df = pd.concat(results)
        else:
            final_df = pd.DataFrame(columns=["user_idx", "item_idx", "relevance"])

        return pandas_to_spark(final_df.astype({"user_idx": int, "item_idx": int}))

from pyspark.sql import Window
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import random

class LSTMRecommender:
    # Packages sequences into tensors
    class SequenceDataset(Dataset):
        def __init__(self, sequences, item_features_tensor):
            self.sequences = sequences
            self.item_features_tensor = item_features_tensor
        
        def __len__(self):
            return len(self.sequences)
    
        def __getitem__(self, idx):
            seq, target = self.sequences[idx]
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            target_tensor = torch.tensor(target, dtype=torch.long)
            feature_tensor = self.item_features_tensor[seq_tensor]
            return seq_tensor, feature_tensor, target_tensor
    # LSTM model
    class LSTMRec(nn.Module):
        def __init__(self, num_items, item_feature_dim, embedding_dim=64, hidden_dim=64, num_layers=2):
            super().__init__()
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.item_feature_proj = nn.Linear(item_feature_dim, 16)
            self.lstm_input_dim = embedding_dim + 16
            self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.5)
            self.fc = nn.Linear(hidden_dim, num_items)
        
        def forward(self, item_ids, item_features):
            embedded = self.item_embedding(item_ids)
            features_proj = self.item_feature_proj(item_features)
            lstm_input = torch.cat([embedded, features_proj], dim=-1)
            lstm_out, _ = self.lstm(lstm_input)
            out = lstm_out[:, -1, :]
            logits = self.fc(out)
            return logits
            
    def __init__(self, seed=42, sequence_length=1, embedding_dim=64, hidden_dim=64, num_epochs=5):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.user_profiles = defaultdict(list)
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Creates user profile sequences using item indices
    def create_dataset(self, log):
        log_pd = log.drop('relevance').toPandas()
        for _, row in log_pd.iterrows():
            self.user_profiles[row['user_idx']].append(row['item_idx'])
        sequence_length = self.sequence_length
        train_sequences = []
        for user, items in self.user_profiles.items():
            if len(items) > sequence_length:
                for i in range(sequence_length, len(items)):
                    seq = items[i-sequence_length:i]
                    target = items[i]
                    train_sequences.append((seq, target))
        return train_sequences
    
    def fit(self, log, user_features=None, item_features=None):
        # Return if empty log
        if log is None or log.count() == 0 or item_features is None:
            return

        # Build feature tensor
        item_features_pd = item_features.toPandas().sort_values('item_idx')
        item_features_pd = pd.get_dummies(item_features_pd, dtype=float)
        feature_cols = [col for col in item_features_pd.columns if col != 'item_idx']
        item_feature_tensor = torch.zeros((item_features_pd['item_idx'].max() + 1, len(feature_cols)))
        for _, row in item_features_pd.iterrows():
            item_feature_tensor[int(row['item_idx'])] = torch.tensor(row[feature_cols].values,dtype=torch.float)

        # Item and feature counts
        total_items = item_feature_tensor.size(0)
        item_feats = len(feature_cols)
        
        # Create user sequences, load data
        train_sequences = self.create_dataset(log)
        train_dataset = self.SequenceDataset(train_sequences, item_feature_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Initialize LSTM Recommender, criterion, optimizer
        self.model = self.LSTMRec(num_items=total_items,item_feature_dim=item_feats, embedding_dim=self.embedding_dim,
                                                   hidden_dim=self.hidden_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Train model, embedding item features
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for sequences, features, targets in train_loader:
                sequences, features, targets = sequences.to(self.device), features.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(sequences, features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

        # Save item feature tensor for future use
        self.item_feature_tensor = item_feature_tensor.to(self.device)
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        # Raise error if model is not trained
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        self.model.eval()
        recommendations = []

        with torch.no_grad():
            # For each user in userbase
            for user in users.toPandas()['user_idx']:
                seq = self.user_profiles.get(user, [])
                seq = seq[-self.sequence_length:]
                # Zero pad sequences if too short
                if len(seq) < self.sequence_length:
                    seq = [0] * (self.sequence_length - len(seq)) + seq

                sequence_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                feature_tensor = self.item_feature_tensor[sequence_tensor].to(self.device)

                logits = self.model(sequence_tensor, feature_tensor)
                probs = torch.softmax(logits, dim=1)
                topk = torch.topk(probs, k=k)
                recommended_item_indices = topk.indices.cpu().numpy().flatten()

                if filter_seen_items:
                    seen_items = set(self.user_profiles.get(user, []))
                    recommended_item_indices = [item for item in recommended_item_indices if item not in seen_items]

                # Item relevance = probability from model * item price
                for item in recommended_item_indices[:k]:
                    #prices = items.select('price').where(items.item_idx == item).rdd.flatMap(lambda x: x).collect()
                    #price = prices[0]
                    relevance = probs[0, item].item()
                    recommendations.append({
                        "user_idx": user,
                        "item_idx": item,
                        "relevance": relevance
                    })

        recommendations_df = pd.DataFrame(recommendations)
        recs = spark.createDataFrame(recommendations_df)
        return recs

class GATRecommender:
    class GATRec(nn.Module):
        def __init__(self, num_nodes, user_feat_dim, item_feat_dim, embedding_dim=256, heads=8):
            super().__init__()
            self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
            in_channels = embedding_dim + user_feat_dim + item_feat_dim
            self.gat1 = GATConv(in_channels, embedding_dim, heads=heads, dropout=0.1)
            self.gat2 = GATConv(embedding_dim * heads, embedding_dim, heads=1, concat=False, dropout=0.1)

        def forward(self, edge_index, user_feats, item_feats):
            x = self.node_embedding.weight

            # Features are floats already shaped [num_nodes, num_features]
            concat_embeds = torch.cat([x, user_feats, item_feats], dim=1)

            x = self.gat1(concat_embeds, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            return x

    def __init__(self, seed=42, embedding_dim=256, num_epochs=10):
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.model = None
        self.user_item_mapping = {}

    def fit(self, log, user_features, item_features):
        import torch.nn.functional as F

        log_pd = log.select("user_idx", "item_idx", "relevance").toPandas()
        user_feat_pd = pd.get_dummies(user_features.toPandas(), dtype=float)
        item_feat_pd = pd.get_dummies(item_features.toPandas(), dtype=float)

        usercols = ['segment_budget', 'segment_mainstream', 'segment_premium']
        itemcols = ['price', 'category_books', 'category_clothing', 'category_electronics', 'category_home']

        unique_users = log_pd['user_idx'].unique()
        unique_items = log_pd['item_idx'].unique()
        user_mapping = {uid: i for i, uid in enumerate(unique_users)}
        item_mapping = {iid: i + len(unique_users) for i, iid in enumerate(unique_items)}
        self.user_item_mapping = {'users': user_mapping, 'items': item_mapping}

        num_nodes = len(unique_users) + len(unique_items)
        num_user_feats = len(usercols)
        num_item_feats = len(itemcols)

        edge_index = []
        pos_pairs = []
        for _, row in log_pd.iterrows():
            u = user_mapping[row['user_idx']]
            i = item_mapping[row['item_idx']]
            edge_index.append([u, i])
            edge_index.append([i, u])
            pos_pairs.append((u, i))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)

        user_feat_tensor = torch.zeros(num_nodes, len(usercols), dtype=torch.long).to(self.device)
        item_feat_tensor = torch.zeros(num_nodes, len(itemcols), dtype=torch.long).to(self.device)

        for _, row in user_feat_pd.iterrows():
            if row['user_idx'] in user_mapping:
                user_feat_tensor[user_mapping[row['user_idx']]] = torch.from_numpy(row[usercols].values).long()

        for _, row in item_feat_pd.iterrows():
            if row['item_idx'] in item_mapping:
                item_feat_tensor[item_mapping[row['item_idx']]] = torch.from_numpy(row[itemcols].values).long()

        self.model = self.GATRec(
            num_nodes=num_nodes,
            user_feat_dim=num_user_feats,
            item_feat_dim=num_item_feats,
            embedding_dim=self.embedding_dim
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        sigmoid = nn.Sigmoid()

        all_item_indices = list(self.user_item_mapping['items'].values())

        for epoch in range(self.num_epochs):
            self.model.train()
            optimizer.zero_grad()

            out = self.model(edge_index, user_feat_tensor, item_feat_tensor)

            loss = 0
            for u, i in pos_pairs:
                u_embed = out[u]
                i_embed = out[i]

                # Hard negative sampling: pick top-N scoring negatives for user u
                with torch.no_grad():
                    item_embeds = out[all_item_indices]  # all item embeddings
                    scores = torch.matmul(u_embed, item_embeds.T)
                    # Mask positive item
                    scores[all_item_indices.index(i)] = -1e9
                    # Top N hardest negatives
                    N = 10
                    top_neg_indices = torch.topk(scores, N).indices.cpu().numpy()
                    j = random.choice(top_neg_indices)
                    j_idx = all_item_indices[j]

                j_embed = out[j_idx]

                pos_score = torch.dot(u_embed, i_embed)
                neg_score = torch.dot(u_embed, j_embed)

                diff = pos_score - neg_score
                diff = torch.clamp(diff, min=-10, max=10)

                loss += -torch.log(sigmoid(diff) + 1e-8)

            loss /= len(pos_pairs)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, BPR Loss: {loss.item():.4f}")

        self.final_embeddings = out

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        self.model.eval()
        recommendations = []
        item_indices = {v: k for k, v in self.user_item_mapping['items'].items()}

        with torch.no_grad():
            for user in users.toPandas()['user_idx']:
                if user not in self.user_item_mapping['users']:
                    continue
                user_idx = self.user_item_mapping['users'][user]
                user_embedding = self.final_embeddings[user_idx]

                item_scores = []
                for item, idx in self.user_item_mapping['items'].items():
                    item_embedding = self.final_embeddings[idx]
                    score = torch.dot(user_embedding, item_embedding).item()
                    item_scores.append((item, score))

                item_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)

                if filter_seen_items:
                    seen_items = set(
                        log.filter(log.user_idx == user).toPandas()['item_idx'].tolist()
                    )
                    item_scores = [pair for pair in item_scores if pair[0] not in seen_items]

                top_k = item_scores[:k]

                for item_id, relevance in top_k:
                    recommendations.append({
                        "user_idx": user,
                        "item_idx": item_id,
                        "relevance": relevance
                    })

        rec_df = pd.DataFrame(recommendations)
        recs = spark.createDataFrame(rec_df)
        return recs

class GCNRecommender:
    """
    Graph Convolutional Network (GCN) Recommender for Checkpoint 3
    
    Implements graph-based collaborative filtering using GCN for link prediction
    on user-item bipartite graphs. Uses message passing to learn user and item
    embeddings that capture collaborative filtering signals.
    
    Key Features:
    - Bipartite user-item graph construction from interaction logs
    - Multi-layer GCN with configurable architecture
    - Link prediction using embedding fusion (dot product)
    - Revenue-aware ranking using expected_revenue = price × link_probability
    - Handles graph updates as new interactions arrive
    - Regularization with embedding L2 penalty and dropout
    """
    
    def __init__(self, embedding_dim=64, num_layers=2, dropout=0.1, learning_rate=0.01, 
                 epochs=50, batch_size=1024, l2_reg=1e-4, seed=None):
        """
        Initialize the GCN recommender.
        
        Args:
            embedding_dim (int): Dimension of user/item embeddings
            num_layers (int): Number of GCN layers (2-4 recommended)
            dropout (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimization
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            l2_reg (float): L2 regularization strength
            seed (int): Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.seed = seed
        
        # Graph and model components
        self.user_mapping = {}  # {original_user_id: graph_node_id}
        self.item_mapping = {}  # {original_item_id: graph_node_id}
        self.reverse_user_mapping = {}  # {graph_node_id: original_user_id}
        self.reverse_item_mapping = {}  # {graph_node_id: original_item_id}
        self.num_users = 0
        self.num_items = 0
        self.item_prices = {}
        
        # PyTorch components (initialized during fit)
        self.model = None
        self.edge_index = None
        self.device = None
        
    def _build_bipartite_graph(self, log):
        """
        Build bipartite user-item graph from interaction log.
        
        Args:
            log: Interaction log DataFrame
            
        Returns:
            tuple: (edge_index, edge_weights, user_mapping, item_mapping)
        """
        import pandas as pd
        import numpy as np
        
        # Handle different log formats
        available_cols = log.columns
        if "price" in available_cols and "response" in available_cols:
            df = log.select("user_idx", "item_idx", "price", "response").toPandas()
            # Only include positive interactions for graph construction
            df = df[df["response"] > 0]
        elif "relevance" in available_cols:
            df = log.select("user_idx", "item_idx", "relevance").toPandas()
            df["price"] = 10.0
            df["response"] = df["relevance"]
            df = df[df["response"] > 0]
        else:
            df = log.select("user_idx", "item_idx").toPandas()
            df["price"] = 10.0
            df["response"] = 1
        
        if df.empty:
            return None, None, {}, {}
        
        # Create user and item mappings
        unique_users = sorted(df["user_idx"].unique())
        unique_items = sorted(df["item_idx"].unique())
        
        self.user_mapping = {user_id: i for i, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: i + len(unique_users) for i, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {i: user_id for user_id, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item_id for item_id, i in self.item_mapping.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        # Store item prices
        for _, row in df.iterrows():
            self.item_prices[row["item_idx"]] = row["price"]
        
        # Build edge list for bipartite graph
        edges = []
        edge_weights = []
        
        for _, row in df.iterrows():
            user_node = self.user_mapping[row["user_idx"]]
            item_node = self.item_mapping[row["item_idx"]]
            weight = row["response"]  # Use response as edge weight
            
            # Add bidirectional edges for bipartite graph
            edges.append([user_node, item_node])
            edges.append([item_node, user_node])
            edge_weights.extend([weight, weight])
        
        edge_index = np.array(edges).T
        edge_weights = np.array(edge_weights)
        
        return edge_index, edge_weights, self.user_mapping, self.item_mapping
    
    def _create_gcn_model(self):
        """Create GCN model using PyTorch (simplified version without PyTorch Geometric)."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for GCN. Install with: pip install torch")
        
        class SimpleGCNModel(nn.Module):
            """
            Simplified GCN implementation without PyTorch Geometric.
            Uses manual message passing for graph convolution.
            """
            def __init__(self, num_nodes, embedding_dim, num_layers, dropout):
                super(SimpleGCNModel, self).__init__()
                self.num_nodes = num_nodes
                self.embedding_dim = embedding_dim
                self.num_layers = num_layers
                self.dropout = dropout
                
                # Node embeddings
                self.embedding = nn.Embedding(num_nodes, embedding_dim)
                
                # Linear layers for each GCN layer
                self.linear_layers = nn.ModuleList()
                for i in range(num_layers):
                    self.linear_layers.append(nn.Linear(embedding_dim, embedding_dim))
                
                # Initialize embeddings
                nn.init.xavier_uniform_(self.embedding.weight)
                for layer in self.linear_layers:
                    nn.init.xavier_uniform_(layer.weight)
            
            def _build_adjacency_matrix(self, edge_index, num_nodes):
                """Build normalized adjacency matrix from edge index."""
                # Create adjacency matrix
                adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
                adj[edge_index[0], edge_index[1]] = 1.0
                
                # Add self-loops
                adj = adj + torch.eye(num_nodes, device=edge_index.device)
                
                # Normalize: D^(-1/2) * A * D^(-1/2)
                degree = adj.sum(dim=1)
                degree_inv_sqrt = torch.pow(degree, -0.5)
                degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0.0
                
                norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
                return norm_adj
            
            def forward(self, edge_index, node_indices=None):
                # Get initial embeddings
                if node_indices is not None:
                    x = self.embedding(node_indices)
                else:
                    x = self.embedding.weight
                
                # Build normalized adjacency matrix
                adj = self._build_adjacency_matrix(edge_index, self.num_nodes)
                
                # Apply GCN layers with manual message passing
                for i, linear in enumerate(self.linear_layers):
                    # Message passing: A * X
                    x = torch.mm(adj, x)
                    
                    # Linear transformation
                    x = linear(x)
                    
                    # Apply activation and dropout (except for last layer)
                    if i < len(self.linear_layers) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                
                return x
            
            def predict_link(self, user_emb, item_emb):
                # Dot product for link prediction
                return torch.sum(user_emb * item_emb, dim=1)
        
        return SimpleGCNModel
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Fit the GCN model on interaction data.
        
        Args:
            log: Interaction log DataFrame
            user_features: User features DataFrame (optional)
            item_features: Item features DataFrame (optional)
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np
        except ImportError:
            print("Warning: PyTorch not available. GCN will use fallback to popularity.")
            self.model = None
            return
        
        if log is None or log.count() == 0:
            self.model = None
            return
        
        # Set device and seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # Build bipartite graph
        edge_index, edge_weights, user_mapping, item_mapping = self._build_bipartite_graph(log)
        
        if edge_index is None:
            self.model = None
            return
        
        # Convert to PyTorch tensors
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        edge_weights = torch.FloatTensor(edge_weights).to(self.device)
        
        # Create model
        total_nodes = self.num_users + self.num_items
        SimpleGCNModel = self._create_gcn_model()
        self.model = SimpleGCNModel(total_nodes, self.embedding_dim, self.num_layers, self.dropout).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        
        # Prepare training data (positive and negative samples)
        pos_edges = []
        for user_id, user_node in self.user_mapping.items():
            for item_id, item_node in self.item_mapping.items():
                # Check if this user-item pair exists in the graph
                user_items = set()
                for i in range(self.edge_index.shape[1]):
                    if self.edge_index[0, i] == user_node and self.edge_index[1, i] >= self.num_users:
                        user_items.add(self.edge_index[1, i].item())
                
                if item_node in user_items:
                    pos_edges.append([user_node, item_node])
        
        pos_edges = torch.LongTensor(pos_edges).to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Forward pass
            node_embeddings = self.model(self.edge_index)
            
            # Positive samples
            if len(pos_edges) > 0:
                user_emb = node_embeddings[pos_edges[:, 0]]
                item_emb = node_embeddings[pos_edges[:, 1]]
                pos_scores = self.model.predict_link(user_emb, item_emb)
                
                # Negative sampling
                neg_edges = []
                for _ in range(len(pos_edges)):
                    user_node = np.random.choice(list(self.user_mapping.values()))
                    item_node = np.random.choice(list(self.item_mapping.values()))
                    neg_edges.append([user_node, item_node])
                
                neg_edges = torch.LongTensor(neg_edges).to(self.device)
                neg_user_emb = node_embeddings[neg_edges[:, 0]]
                neg_item_emb = node_embeddings[neg_edges[:, 1]]
                neg_scores = self.model.predict_link(neg_user_emb, neg_item_emb)
                
                # BPR loss (Bayesian Personalized Ranking)
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.model.eval()
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate graph-based recommendations using GCN.
        
        Args:
            log: Current interaction log
            k: Number of items to recommend per user
            users: Users DataFrame
            items: Items DataFrame
            user_features: User features (optional)
            item_features: Item features (optional)
            filter_seen_items: Whether to exclude already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, relevance columns
        """
        import pandas as pd
        import numpy as np
        from sim4rec.utils import pandas_to_spark
        from pyspark.sql.types import LongType
        import pyspark.sql.functions as sf

        # Fallback to PopularityRecommender if no model
        if self.model is None:
            from recommender_analysis_visualization import PopularityRecommender
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, 
                                 filter_seen_items=filter_seen_items)

        try:
            import torch
        except ImportError:
            # Fallback if PyTorch not available
            from recommender_analysis_visualization import PopularityRecommender
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, 
                                 filter_seen_items=filter_seen_items)

        user_list = users.select("user_idx").toPandas()["user_idx"].tolist()
        item_list = items.select("item_idx", "price").toPandas()
        recommendations = []
        
        # Get current embeddings
        with torch.no_grad():
            node_embeddings = self.model(self.edge_index)
        
        # Get seen items for filtering
        seen_items_per_user = {}
        if filter_seen_items and log is not None:
            log_df = log.select("user_idx", "item_idx").toPandas()
            for user_id, group in log_df.groupby("user_idx"):
                seen_items_per_user[user_id] = set(group["item_idx"].tolist())
        
        for user_id in user_list:
            # Check if user exists in graph
            if user_id not in self.user_mapping:
                # Cold start: fallback to popularity
                from recommender_analysis_visualization import PopularityRecommender
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), 
                                         items, user_features, item_features, 
                                         filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recommendations.extend(pop_recs_pd.to_dict("records"))
                continue
            
            user_node = self.user_mapping[user_id]
            user_emb = node_embeddings[user_node].unsqueeze(0)  # Add batch dimension
            
            candidate_items = []
            seen_items = seen_items_per_user.get(user_id, set())
            
            for _, item_row in item_list.iterrows():
                item_id = int(item_row['item_idx'])
                price = item_row['price']
                
                # Skip if already seen
                if filter_seen_items and item_id in seen_items:
                    continue
                
                # Check if item exists in graph
                if item_id not in self.item_mapping:
                    continue
                
                item_node = self.item_mapping[item_id]
                item_emb = node_embeddings[item_node].unsqueeze(0)  # Add batch dimension
                
                # Predict link probability
                with torch.no_grad():
                    link_score = self.model.predict_link(user_emb, item_emb).item()
                    link_probability = torch.sigmoid(torch.tensor(link_score)).item()
                
                # Calculate expected revenue = probability × price
                expected_revenue = link_probability * price
                
                candidate_items.append({
                    "user_idx": user_id,
                    "item_idx": item_id,
                    "relevance": expected_revenue
                })
            
            # Rank by expected revenue and select top-k
            candidate_items.sort(key=lambda x: x["relevance"], reverse=True)
            top_recommendations = candidate_items[:k]
            
            if not top_recommendations:
                # No candidates: fallback to popularity
                from recommender_analysis_visualization import PopularityRecommender
                pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
                pop_rec.fit(log, user_features, item_features)
                pop_recs = pop_rec.predict(log, k, users.filter(sf.col("user_idx") == user_id), 
                                         items, user_features, item_features, 
                                         filter_seen_items=filter_seen_items)
                pop_recs_pd = pop_recs.select("user_idx", "item_idx", "relevance").toPandas()
                recommendations.extend(pop_recs_pd.to_dict("records"))
            else:
                recommendations.extend(top_recommendations)
        
        # Convert to Spark DataFrame with proper types
        if not recommendations:
            from recommender_analysis_visualization import PopularityRecommender
            pop_rec = PopularityRecommender(alpha=1.0, seed=self.seed)
            pop_rec.fit(log, user_features, item_features)
            return pop_rec.predict(log, k, users, items, user_features, item_features, 
                                 filter_seen_items=filter_seen_items)
        
        recs_df = pd.DataFrame(recommendations)
        recs_df["user_idx"] = recs_df["user_idx"].astype(np.int64)
        recs_df["item_idx"] = recs_df["item_idx"].astype(np.int64)
        
        result = pandas_to_spark(recs_df)
        result = result.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        result = result.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
        
        return result
    
# Cell: Data Exploration Functions
"""
## Data Exploration Functions
These functions help us understand the generated synthetic data.
"""

def explore_user_data(users_df):
    """
    Explore user data distributions and characteristics.
    
    Args:
        users_df: DataFrame containing user data
    """
    print("=== User Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of users: {users_df.count()}")
    
    # User segments distribution
    segment_counts = users_df.groupBy("segment").count().toPandas()
    print("\nUser Segments Distribution:")
    for _, row in segment_counts.iterrows():
        print(f"  {row['segment']}: {row['count']} users ({row['count']/users_df.count()*100:.1f}%)")
    
    # Plot user segments
    plt.figure(figsize=(10, 6))
    plt.pie(segment_counts['count'], labels=segment_counts['segment'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('User Segments Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('user_segments_distribution.png')
    print("User segments visualization saved to 'user_segments_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    users_pd = users_df.toPandas()
    
    # Analyze user feature distributions
    feature_cols = [col for col in users_pd.columns if col.startswith('user_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for segment in users_pd['segment'].unique():
                segment_data = users_pd[users_pd['segment'] == segment]
                plt.hist(segment_data[feature], alpha=0.5, bins=20, label=segment)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('user_feature_distributions.png')
        print("User feature distributions saved to 'user_feature_distributions.png'")
        
        # Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        corr = users_pd[feature_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                    square=True, linewidths=.5, annot=False, fmt='.2f')
        plt.title('User Feature Correlations')
        plt.tight_layout()
        plt.savefig('user_feature_correlations.png')
        print("User feature correlations saved to 'user_feature_correlations.png'")


def explore_item_data(items_df):
    """
    Explore item data distributions and characteristics.
    
    Args:
        items_df: DataFrame containing item data
    """
    print("\n=== Item Data Exploration ===")
    
    # Get basic statistics
    print(f"Total number of items: {items_df.count()}")
    
    # Item categories distribution
    category_counts = items_df.groupBy("category").count().toPandas()
    print("\nItem Categories Distribution:")
    for _, row in category_counts.iterrows():
        print(f"  {row['category']}: {row['count']} items ({row['count']/items_df.count()*100:.1f}%)")
    
    # Plot item categories
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts['count'], labels=category_counts['category'], autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Item Categories Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('item_categories_distribution.png')
    print("Item categories visualization saved to 'item_categories_distribution.png'")
    
    # Convert to pandas for easier feature analysis
    items_pd = items_df.toPandas()
    
    # Analyze price distribution
    if 'price' in items_pd.columns:
        plt.figure(figsize=(14, 6))
        
        # Overall price distribution
        plt.subplot(1, 2, 1)
        plt.hist(items_pd['price'], bins=30, alpha=0.7)
        plt.title('Overall Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Count')
        
        # Price by category
        plt.subplot(1, 2, 2)
        for category in items_pd['category'].unique():
            category_data = items_pd[items_pd['category'] == category]
            plt.hist(category_data['price'], alpha=0.5, bins=20, label=category)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('item_price_distributions.png')
        print("Item price distributions saved to 'item_price_distributions.png'")
    
    # Analyze item feature distributions
    feature_cols = [col for col in items_pd.columns if col.startswith('item_attr_')]
    if len(feature_cols) > 0:
        # Take a sample of feature columns if there are many
        sample_features = feature_cols[:min(5, len(feature_cols))]
        
        # Plot histograms for sample features
        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(sample_features):
            plt.subplot(2, 3, i+1)
            for category in items_pd['category'].unique():
                category_data = items_pd[items_pd['category'] == category]
                plt.hist(category_data[feature], alpha=0.5, bins=20, label=category)
            plt.title(f'Distribution of {feature}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            if i == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig('item_feature_distributions.png')
        print("Item feature distributions saved to 'item_feature_distributions.png'")


def explore_interactions(history_df, users_df, items_df):
    """
    Explore interaction patterns between users and items.
    
    Args:
        history_df: DataFrame containing interaction history
        users_df: DataFrame containing user data
        items_df: DataFrame containing item data
    """
    print("\n=== Interaction Data Exploration ===")
    
    # Get basic statistics
    total_interactions = history_df.count()
    total_users = users_df.count()
    total_items = items_df.count()
    
    print(f"Total interactions: {total_interactions}")
    print(f"Interaction density: {total_interactions / (total_users * total_items) * 100:.4f}%")
    
    # Users with interactions
    users_with_interactions = history_df.select("user_idx").distinct().count()
    print(f"Users with at least one interaction: {users_with_interactions} ({users_with_interactions/total_users*100:.1f}%)")
    
    # Items with interactions
    items_with_interactions = history_df.select("item_idx").distinct().count()
    print(f"Items with at least one interaction: {items_with_interactions} ({items_with_interactions/total_items*100:.1f}%)")
    
    # Distribution of interactions per user
    interactions_per_user = history_df.groupBy("user_idx").count().toPandas()
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(interactions_per_user['count'], bins=20)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    
    # Distribution of interactions per item
    interactions_per_item = history_df.groupBy("item_idx").count().toPandas()
    
    plt.subplot(1, 2, 2)
    plt.hist(interactions_per_item['count'], bins=20)
    plt.title('Distribution of Interactions per Item')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Items')
    
    plt.tight_layout()
    plt.savefig('interaction_distributions.png')
    print("Interaction distributions saved to 'interaction_distributions.png'")
    
    # Analyze relevance distribution
    if 'relevance' in history_df.columns:
        relevance_dist = history_df.groupBy("relevance").count().toPandas()
        
        plt.figure(figsize=(10, 6))
        plt.bar(relevance_dist['relevance'].astype(str), relevance_dist['count'])
        plt.title('Distribution of Relevance Scores')
        plt.xlabel('Relevance Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('relevance_distribution.png')
        print("Relevance distribution saved to 'relevance_distribution.png'")
    
    # If we have user segments and item categories, analyze cross-interactions
    if 'segment' in users_df.columns and 'category' in items_df.columns:
        # Join with user segments and item categories
        interaction_analysis = history_df.join(
            users_df.select('user_idx', 'segment'),
            on='user_idx'
        ).join(
            items_df.select('item_idx', 'category'),
            on='item_idx'
        )
        
        # Count interactions by segment and category
        segment_category_counts = interaction_analysis.groupBy('segment', 'category').count().toPandas()
        
        # Create a pivot table
        pivot_table = segment_category_counts.pivot(index='segment', columns='category', values='count').fillna(0)
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='g', cmap='viridis')
        plt.title('Interactions Between User Segments and Item Categories')
        plt.tight_layout()
        plt.savefig('segment_category_interactions.png')
        print("Segment-category interactions saved to 'segment_category_interactions.png'")


# Cell: Recommender Analysis Function
"""
## Recommender System Analysis
This is the main function to run analysis of different recommender systems and visualize the results.
"""

def run_recommender_analysis():
    """
    Run an analysis of different recommender systems and visualize the results.
    This function creates a synthetic dataset, performs EDA, evaluates multiple recommendation
    algorithms using train-test split, and visualizes the performance metrics.
    """
    # Create a smaller dataset for experimentation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000  # Reduced from 10,000
    config['data_generation']['n_items'] = 200   # Reduced from 1,000
    config['data_generation']['seed'] = 42       # Fixed seed for reproducibility
    
    # Get train-test split parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running train-test simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate user data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    # Generate item data
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    # Generate initial interaction history
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Cell: Exploratory Data Analysis
    """
    ## Exploratory Data Analysis
    Let's explore the generated synthetic data before running the recommenders.
    """
    
    # Perform exploratory data analysis on the generated data
    print("\n=== Starting Exploratory Data Analysis ===")
    explore_user_data(users_df)
    explore_item_data(items_df)
    explore_interactions(history_df, users_df, items_df)
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Cell: Setup and Run Recommenders
    """
    ## Recommender Systems Comparison
    Now we'll set up and evaluate different recommendation algorithms.
    """
    
    # Initialize recommenders to compare
    recommenders = [
        RandomRecommender(seed=42),
        # PopularityRecommender(alpha=1.0, seed=42),
        # ContentBasedRecommender(similarity_threshold=0.0, seed=42),
        # MyRecommender(seed=42),  # Add your custom recommender here
        # DecisionTreeRecommender(seed=42, max_depth=5, min_samples_leaf=1, min_samples_split=2, ccp_alpha=0.0),
        LogisticRegressionRecommender(),
        # KNNRecommender(k=5, seed=42),
        TransformerRecommender(),
        AutoRegressiveRecommender(),
        LSTMRecommender(),
        GATRecommender(),
        GCNRecommender(),
        Node2VecRecommender()
    ] 
    recommender_names = [
        # "Random", 
        # "Popularity", 
        # "ContentBased", 
        # "MyRecommender", 
        # "DecisionTree", 
        "LogisticRegression",
        # "KNN", 
        "Transformer", 
        "AutoRegressiveRecommender",
        "LSTMRecommender",
        "GATRecommender",
        "GCNRecommender",
        "Node2Vec"
    ]
    
    # Initialize recommenders with initial history
    for recommender in recommenders:
        recommender.fit(log=data_generator.history_df, 
                        user_features=users_df, 
                        item_features=items_df)
    
    # Evaluate each recommender separately using train-test split
    results = []
    
    for name, recommender in zip(recommender_names, recommenders):
        print(f"\nEvaluating {name}:")
        
        # Clean up any existing simulator data directory for this recommender
        simulator_data_dir = f"simulator_train_test_data_{name}"
        if os.path.exists(simulator_data_dir):
            shutil.rmtree(simulator_data_dir)
            print(f"Removed existing simulator data directory: {simulator_data_dir}")
        
        # Initialize simulator
        simulator = CompetitionSimulator(
            user_generator=user_generator,
            item_generator=item_generator,
            data_dir=simulator_data_dir,
            log_df=data_generator.history_df,  # PySpark DataFrames don't have copy method
            conversion_noise_mean=config['simulation']['conversion_noise_mean'],
            conversion_noise_std=config['simulation']['conversion_noise_std'],
            spark_session=spark,
            seed=config['data_generation']['seed']
        )
        
        # Run simulation with train-test split
        train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
            recommender=recommender,
            train_iterations=train_iterations,
            test_iterations=test_iterations,
            user_frac=config['simulation']['user_fraction'],
            k=config['simulation']['k'],
            filter_seen_items=config['simulation']['filter_seen_items'],
            retrain=config['simulation']['retrain']
        )
        
        # Calculate average metrics
        train_avg_metrics = {}
        for metric_name in train_metrics[0].keys():
            values = [metrics[metric_name] for metrics in train_metrics]
            train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
        
        test_avg_metrics = {}
        for metric_name in test_metrics[0].keys():
            values = [metrics[metric_name] for metrics in test_metrics]
            test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
        
        # Store results
        results.append({
            "name": name,
            "train_total_revenue": sum(train_revenue),
            "test_total_revenue": sum(test_revenue),
            "train_avg_revenue": np.mean(train_revenue),
            "test_avg_revenue": np.mean(test_revenue),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_revenue": train_revenue,
            "test_revenue": test_revenue,
            **train_avg_metrics,
            **test_avg_metrics
        })
        
        # Print summary for this recommender
        print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
        print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
        performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
        print(f"  Performance Change: {performance_change:.2f}%")
    
    # Convert to DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
    
    # Print summary table
    print("\nRecommender Evaluation Results (sorted by test revenue):")
    summary_cols = ["name", "train_total_revenue", "test_total_revenue", 
                   "train_avg_revenue", "test_avg_revenue",
                   "train_precision_at_k", "test_precision_at_k",
                   "train_ndcg_at_k", "test_ndcg_at_k",
                   "train_mrr", "test_mrr",
                   "train_discounted_revenue", "test_discounted_revenue"]
    summary_cols = [col for col in summary_cols if col in results_df.columns]
    
    print(results_df[summary_cols].to_string(index=False))
    
    # Cell: Results Visualization
    """
    ## Results Visualization
    Now we'll visualize the performance of the different recommenders.
    """
    
    # Generate comparison plots
    visualize_recommender_performance(results_df, recommender_names)
    
    # Generate detailed metrics visualizations
    visualize_detailed_metrics(results_df, recommender_names)
    
    return results_df


# Cell: Performance Visualization Functions
"""
## Performance Visualization Functions
These functions create visualizations for comparing recommender performance.
"""

def visualize_recommender_performance(results_df, recommender_names):
    """
    Visualize the performance of recommenders in terms of revenue and key metrics.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    plt.figure(figsize=(16, 16))
    
    # Plot total revenue comparison
    plt.subplot(3, 2, 1)
    x = np.arange(len(recommender_names))
    width = 0.35
    plt.bar(x - width/2, results_df['train_total_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_total_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Total Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot average revenue per iteration
    plt.subplot(3, 2, 2)
    plt.bar(x - width/2, results_df['train_avg_revenue'], width, label='Training')
    plt.bar(x + width/2, results_df['test_avg_revenue'], width, label='Testing')
    plt.xlabel('Recommender')
    plt.ylabel('Avg Revenue per Iteration')
    plt.title('Average Revenue Comparison')
    plt.xticks(x, results_df['name'])
    plt.legend()
    
    # Plot discounted revenue comparison (if available)
    plt.subplot(3, 2, 3)
    if 'train_discounted_revenue' in results_df.columns and 'test_discounted_revenue' in results_df.columns:
        plt.bar(x - width/2, results_df['train_discounted_revenue'], width, label='Training')
        plt.bar(x + width/2, results_df['test_discounted_revenue'], width, label='Testing')
        plt.xlabel('Recommender')
        plt.ylabel('Avg Discounted Revenue')
        plt.title('Discounted Revenue Comparison')
        plt.xticks(x, results_df['name'])
        plt.legend()
    
    # Plot revenue trajectories
    plt.subplot(3, 2, 4)
    markers = ['o', 's', 'D', '^']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, name in enumerate(results_df['name']):
        # Combined train and test trajectories
        train_revenue = results_df.iloc[i]['train_revenue']
        test_revenue = results_df.iloc[i]['test_revenue']
        
        # Check if revenue is a scalar (numpy.float64) or a list/array
        if isinstance(train_revenue, (float, np.float64, np.float32, int, np.integer)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, np.float32, int, np.integer)):
            test_revenue = [test_revenue]
            
        iterations = list(range(len(train_revenue))) + list(range(len(test_revenue)))
        revenues = train_revenue + test_revenue
        
        plt.plot(iterations, revenues, marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)], label=name)
        
        # Add a vertical line to separate train and test
        if i == 0:  # Only add the line once
            plt.axvline(x=len(train_revenue)-0.5, color='k', linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Trajectory (Training → Testing)')
    plt.legend()
    
    # Plot ranking metrics comparison - Training
    plt.subplot(3, 2, 5)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'train_{m}' in results_df.columns]
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'train_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Training Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    # Plot ranking metrics comparison - Testing
    plt.subplot(3, 2, 6)
    
    # Select metrics to include
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    ranking_metrics = [m for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    # Get best-performing model
    best_model_idx = results_df['test_total_revenue'].idxmax()
    best_model_name = results_df.iloc[best_model_idx]['name']
    
    # Create bar groups
    bar_positions = np.arange(len(ranking_metrics))
    bar_width = 0.8 / len(results_df)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['name']
        offsets = (i - len(results_df)/2 + 0.5) * bar_width
        metric_values = [row[f'test_{m}'] for m in ranking_metrics]
        plt.bar(bar_positions + offsets, metric_values, bar_width, label=model_name, 
                color=colors[i % len(colors)],
                alpha=0.7 if model_name != best_model_name else 1.0)
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Ranking Metrics Comparison (Test Phase)')
    plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in ranking_metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('recommender_performance_comparison.png')
    print("\nPerformance visualizations saved to 'recommender_performance_comparison.png'")


def visualize_detailed_metrics(results_df, recommender_names):
    """
    Create detailed visualizations for each metric and recommender.
    
    Args:
        results_df: DataFrame with evaluation results
        recommender_names: List of recommender names
    """
    # Create a figure for metric trajectories
    plt.figure(figsize=(16, 16))
    
    # Get all available metrics
    all_metrics = []
    if len(results_df) > 0 and 'train_metrics' in results_df.columns:
        first_train_metrics = results_df.iloc[0]['train_metrics'][0]
        all_metrics = list(first_train_metrics.keys())
    
    # Select key metrics to visualize
    key_metrics = ['revenue', 'discounted_revenue', 'precision_at_k', 'ndcg_at_k', 'mrr', 'hit_rate']
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Plot metric trajectories for each key metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', 'D', '^']
    
    for i, metric in enumerate(key_metrics):
        if i < 6:  # Limit to 6 metrics to avoid overcrowding
            plt.subplot(3, 2, i+1)
            
            for j, name in enumerate(results_df['name']):
                row = results_df[results_df['name'] == name].iloc[0]
                
                # Get metric values for training phase
                train_values = []
                for train_metric in row['train_metrics']:
                    if metric in train_metric:
                        train_values.append(train_metric[metric])
                
                # Get metric values for testing phase
                test_values = []
                for test_metric in row['test_metrics']:
                    if metric in test_metric:
                        test_values.append(test_metric[metric])
                
                # Plot training phase
                plt.plot(range(len(train_values)), train_values, 
                         marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='-', label=f"{name} (train)")
                
                # Plot testing phase
                plt.plot(range(len(train_values), len(train_values) + len(test_values)), 
                         test_values, marker=markers[j % len(markers)], 
                         color=colors[j % len(colors)],
                         linestyle='--', label=f"{name} (test)")
                
                # Add a vertical line to separate train and test
                if j == 0:  # Only add the line once
                    plt.axvline(x=len(train_values)-0.5, color='k', 
                                linestyle='--', alpha=0.3, label='Train/Test Split')
            
            # Get metric info from EVALUATION_METRICS
            if metric in EVALUATION_METRICS:
                metric_info = EVALUATION_METRICS[metric]
                metric_name = metric_info['name']
                plt.title(f"{metric_name} Trajectory")
            else:
                plt.title(f"{metric.replace('_', ' ').title()} Trajectory")
            
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            
            # Add legend to the last plot only to avoid cluttering
            if i == len(key_metrics) - 1 or i == 5:
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('recommender_metrics_trajectories.png')
    print("Detailed metrics visualizations saved to 'recommender_metrics_trajectories.png'")
    
    # Create a correlation heatmap of metrics
    plt.figure(figsize=(14, 12))
    
    # Extract metrics columns
    metric_cols = [col for col in results_df.columns if col.startswith('train_') or col.startswith('test_')]
    metric_cols = [col for col in metric_cols if not col.endswith('_metrics') and not col.endswith('_revenue')]
    
    if len(metric_cols) > 1:
        correlation_df = results_df[metric_cols].corr()
        
        # Plot heatmap
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig('metrics_correlation_heatmap.png')
        print("Metrics correlation heatmap saved to 'metrics_correlation_heatmap.png'")


def calculate_discounted_cumulative_gain(recommendations, k=5, discount_factor=0.85):
    """
    Calculate the Discounted Cumulative Gain for recommendations.
    
    Args:
        recommendations: DataFrame with recommendations (must have relevance column)
        k: Number of items to consider
        discount_factor: Factor to discount gains by position
        
    Returns:
        float: Average DCG across all users
    """
    # Group by user and calculate per-user DCG
    user_dcg = []
    for user_id, user_recs in recommendations.groupBy("user_idx").agg(
        sf.collect_list(sf.struct("relevance", "rank")).alias("recommendations")
    ).collect():
        # Sort by rank
        user_rec_list = sorted(user_id.recommendations, key=lambda x: x[1])
        
        # Calculate DCG
        dcg = 0
        for i, (rel, _) in enumerate(user_rec_list[:k]):
            # Apply discount based on position
            dcg += rel * (discount_factor ** i)
        
        user_dcg.append(dcg)
    
    # Return average DCG across all users
    return np.mean(user_dcg) if user_dcg else 0.0


# Cell: Main execution
"""
## Run the Analysis
When you run this notebook, it will perform the full analysis and visualization.
"""

if __name__ == "__main__":
    results = run_recommender_analysis() 
