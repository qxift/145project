import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sim4rec.utils import pandas_to_spark
from pyspark.sql.types import LongType
import pyspark.sql.functions as sf
from sample_recommenders import PopularityRecommender

class MyRecommender:
    def __init__(self, walk_length=8, num_walks=40, embedding_dim=32, window_size=4, epochs=3, seed=42):
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.seed = seed
        self.user_embeddings = {}
        self.item_embeddings = {}

    def fit(self, log, user_features=None, item_features=None):
        df = log.select("user_idx", "item_idx").toPandas()
        if df.empty:
            return

        offset = df["user_idx"].max() + 1
        df["item_idx"] += offset

        G = nx.Graph()
        G.add_edges_from(df.values)

        walks = []
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_walks):
            nodes = list(G.nodes())
            rng.shuffle(nodes)
            for node in nodes:
                walk = [node]
                while len(walk) < self.walk_length:
                    neighbors = list(G.neighbors(walk[-1]))
                    if not neighbors:
                        break
                    walk.append(rng.choice(neighbors))
                walks.append([str(n) for n in walk])

        w2v = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,
            epochs=self.epochs,
            seed=self.seed,
            workers=1
        )

        self.user_embeddings = {
            int(k): w2v.wv[k] for k in w2v.wv.key_to_index if int(k) < offset
        }
        self.item_embeddings = {
            int(k) - offset: w2v.wv[k] for k in w2v.wv.key_to_index if int(k) >= offset
        }

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        items_df = items.select("item_idx", "price").toPandas()
        if items_df.empty:
            return pandas_to_spark(pd.DataFrame(columns=["user_idx", "item_idx", "relevance"]))

        item_vecs = np.stack([
            self.item_embeddings.get(i, np.zeros(self.embedding_dim)) for i in items_df["item_idx"]
        ])
        seen_df = log.select("user_idx", "item_idx").toPandas() if filter_seen_items else pd.DataFrame()
        results = []

        for user_id in users.select("user_idx").toPandas()["user_idx"]:
            user_vec = self.user_embeddings.get(user_id)
            if user_vec is None:
                continue

            sims = cosine_similarity(user_vec.reshape(1, -1), item_vecs).flatten()
            item_scores = items_df.copy()
            item_scores["prob"] = 1 / (1 + np.exp(-sims))  # sigmoid smoothing
            item_scores["relevance"] = item_scores["prob"] * item_scores["price"]

            if filter_seen_items:
                seen_items = seen_df[seen_df["user_idx"] == user_id]["item_idx"].values
                item_scores = item_scores[~item_scores["item_idx"].isin(seen_items)]

            topk = item_scores.nlargest(k, "relevance")
            topk["user_idx"] = user_id
            results.append(topk[["user_idx", "item_idx", "relevance"]])

        if not results:
            return pandas_to_spark(pd.DataFrame(columns=["user_idx", "item_idx", "relevance"]))

        final_df = pd.concat(results)
        result = pandas_to_spark(final_df.astype({"user_idx": int, "item_idx": int}))
        return result.withColumn("user_idx", sf.col("user_idx").cast(LongType())) \
                     .withColumn("item_idx", sf.col("item_idx").cast(LongType()))
