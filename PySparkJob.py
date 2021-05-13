import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F


def process(spark, input_file, target_path):
    df = spark.read.parquet(input_file)

    views = df.filter(df.event == 'view').select('ad_id', 'event').groupBy('ad_id').count().withColumnRenamed('count',
                                                                                                              'views').withColumnRenamed(
        'ad_id', 'ad_id0')
    clicks = df.filter(df.event == 'click').select('ad_id', 'event').groupBy('ad_id').count().withColumnRenamed('count',
                                                                                                                'clicks').withColumnRenamed(
        'ad_id', 'ad_id1')
    ctr = views.join(clicks, views.ad_id0 == clicks.ad_id1, 'outer')
    ctr = ctr.fillna({'clicks': 0})
    ctr = ctr.withColumn('CTR', ctr.clicks / ctr.views)
    ctr = ctr.drop('ad_id1', 'views', 'clicks')

    days = df.select('ad_id', 'date').distinct().groupBy('ad_id').count().withColumnRenamed('count',
                                                                                            'day_count').withColumnRenamed(
        'ad_id', 'to_drop')

    df = df.withColumn('is_cpm', col('ad_cost_type') == 'CPM')
    df = df.withColumn('is_cpc', col('ad_cost_type') == 'CPC')
    df = df.join(days, df.ad_id == days.to_drop, 'outer')
    df = df.join(ctr, df.ad_id == ctr.ad_id0, 'outer')
    df = df.drop('to_drop', 'ad_id0')
    df = df.select('ad_id', 'target_audience_count', 'has_video', df.is_cpm.cast('int'), df.is_cpc.cast('int'),
                   'ad_cost', 'day_count', 'CTR').distinct().orderBy(F.desc('target_audience_count'))
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