import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    #Ваш код
    
    # TRAIN
    train_data = spark.read.parquet(train_data)
    feature = VectorAssembler(inputCols=train_data.columns[1:], outputCol='features')
    train_data = feature.transform(train_data)
    
    # TEST
    test_data = spark.read.parquet(test_data)
    feature = VectorAssembler(inputCols=test_data.columns[1:], outputCol='features')
    test_data = feature.transform(test_data)

    
    regressions = [DecisionTreeRegressor, RandomForestRegressor, GBTRegressor]
    best_regression = {}

    
    def dt(type_regression, action):
        # Decision tree regression
        dt = type_regression(featuresCol="features", labelCol='ctr')
        
        paramGrid = ParamGridBuilder()\
            .addGrid(dt.maxDepth, [i for i in range(1, 3)])\
            .addGrid(dt.maxBins, [i for i in range(2, 4)])\
            .build()
        
        tvs = TrainValidationSplit(estimator=dt,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(labelCol='ctr'),
                           trainRatio=0.8)
        
        model = dt.fit(train_data)
        
        dt = type_regression(featuresCol='features', labelCol='CTR', maxBins=model.bestModel._java_obj.getMaxBins(), maxDepth=model.bestModel._java_obj.getMaxDepth())
       

        predictions = model.transform(test_data)

        evaluator = RegressionEvaluator(
        labelCol="ctr", predictionCol="prediction", metricName="rmse")

        
        rmse = evaluator.evaluate(predictions)
        if action == 0:
            print(f'For {type_regression} RMSE = {rmse}')
            print(f'maxBins = {model.bestModel._java_obj.getMaxBins()}')
            print(f'maxDepth = {model.bestModel._java_obj.getMaxDepth()}')
            return rmse
        else:
            model.write().overwrite().save(MODEL_PATH)
            print('Model saved!')
    
    for i in regressions:
        best_regression[i] = dt(i, 0)
        
    sort_regressions = list(best_regression.items())
    sort_regressions.sort(key = lambda i: i[1])
    
    dt(sort_regressions[0][0], 1)

    
    
def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
