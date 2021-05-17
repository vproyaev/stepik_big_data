import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.sql import functions as F


def process(spark, input_file, target_path):
    df = spark.read.parquet(input_file)

    ctr = df.groupBy('ad_id').agg((sum(F.when(F.col('event') == 'click', 1).otherwise(0)).alias('click') / sum(F.when(F.col('event') == 'view', 1).otherwise(0)).alias('view')).alias('CTR')).withColumnRenamed('ad_id', 'ad_id1')
    days = df.select('ad_id', 'date').distinct().groupBy('ad_id').count().withColumnRenamed('count', 'day_count').withColumnRenamed('ad_id', 'ad_id2')

    df = df.withColumn('is_cpm', (col('ad_cost_type') == 'CPM').cast('int')).withColumn('is_cpc', (col('ad_cost_type') == 'CPC').cast('int'))

    df = df.join(days, df.ad_id == days.ad_id2, 'outer')
    df = df.join(ctr, df.ad_id == ctr.ad_id1, 'outer')
    df = df.drop('ad_id1', 'ad_id2')

    df = df.select('ad_id', 'target_audience_count', 'has_video', 'is_cpm', 'is_cpc', 'ad_cost', 'day_count', 'CTR').distinct().orderBy(F.desc('target_audience_count'))

    df = df.fillna({'CTR': 0})

    df = df.randomSplit([0.75, 0.25])
    df[0].coalesce(4).write.parquet(f'{target_path}/train')
    df[1].coalesce(4).write.parquet(f'{target_path}/test')


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
