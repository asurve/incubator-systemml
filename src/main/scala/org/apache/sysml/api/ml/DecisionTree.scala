/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.api.ml

import org.apache.spark.rdd.RDD
import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.ml.{Estimator, Model, Pipeline}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.param._
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._

object DecisionTree {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "decision-tree.dml"
}

class DecisionTree(override val uid: String, val sc: SparkContext) extends Estimator[DecisionTreeModel]
  with HasImpurity with HasBins with HasDepth with HasNumLeaf with HasNumSamples with HasRCol with BaseSystemMLClassifier {

  def setImpurity(value: String) = set(impurity, value)
  def setBins(value: Int) = set(bins, value)
  def setDepth(value: Int) = set(depth, value)
  def setNumLeaf(value: Int) = set(numLeaf, value)
  def setNumSamples(value: Int) = set(numSamples, value)
  def setRCol(value: String) = set(rCol, value)


  override def copy(extra: ParamMap): Estimator[DecisionTreeModel] = {
    val that = new DecisionTree(uid, sc)
    copyValues(that, extra)
  }

  // Note: will update the y_mb as this will be called by Python mllearn
  def fit(X_mb: MatrixBlock, y_mb: MatrixBlock): DecisionTreeModel = {
    val ret = baseFit(X_mb, y_mb, sc)
    new DecisionTreeModel("decisionTree")(ret, sc)
  }

  def fit(df: ScriptsUtils.SparkDataType): DecisionTreeModel = {
    val ret = baseFit(df, sc)
    new DecisionTreeModel("decisionTree")(ret, sc)
  }

  def getTrainingScript(isSingleNode:Boolean):(Script, String, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(DecisionTree.scriptPath))
      .in("$X", " ")
      .in("$Y", " ")
      .in("$impurity", getImpurity)
      .in("$bins", getBins)
      .in("$depth", getDepth)
      .in("$numLeaf", getNumLeaf)
      .in("$numSamples", getNumSamples)
      .in("$rCol", getRCol)
      .out("M")
    (script, "X", "Y_bin")
  }
}


object DecisionTreeModel {
  final val scriptPath = "scripts" + File.separator + "algorithms" + File.separator + "decision-tree-predict.dml"
}

class DecisionTreeModel(override val uid: String)
                       (val mloutput: MLResults, val sc: SparkContext)
  extends Model[DecisionTreeModel] with BaseSystemMLClassifierModel {
  
  override def copy(extra: ParamMap): DecisionTreeModel = {
    val that = new DecisionTreeModel(uid)(mloutput, sc)
    copyValues(that, extra)
  }
  
  def getPredictionScript(mloutput: MLResults, isSingleNode:Boolean): (Script, String)  = {
    val script = dml(ScriptsUtils.getDMLScript(DecisionTreeModel.scriptPath))
      .in("$X", " ")
      .in("$M", " ")
      .in("$P", " ")
      .out("Y_predicted")

    val YPredicted = mloutput.getDataFrame("Y_predicted")
    (script, "X_test")
  }

  def transform(X: MatrixBlock): MatrixBlock = baseTransform(X, mloutput, sc, "Y_predicted")
  def transform(df: ScriptsUtils.SparkDataType): DataFrame = baseTransform(df, mloutput, sc, "Y_predicted")

}


/**
  * Example code for Decision Tree
  */

/*
object DecisionTreeExample {

  import org.apache.spark.{SparkConf, SparkContext}
  import org.apache.spark.sql.types._
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.regression.LabeledPoint

  def main(args: Array[String]) = {
    val sparkConf: SparkConf = new SparkConf();
    val sc: SparkContext = new SparkContext("local", "TestLocal", sparkConf);
    val sqlContext = new org.apache.spark.sql.SQLContext(sc);

    import sqlContext.implicits._
    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.5, 2.2)),
      LabeledPoint(2.0, Vectors.dense(1.6, 0.8, 3.6)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 2.3))))
    val lr = new DecisionTree("decisionTree", sc)
    val lrmodel = lr.fit(training.toDF)

    val testing = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.5, 2.2)),
      LabeledPoint(2.0, Vectors.dense(1.6, 0.8, 3.6)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 2.3))))

    lrmodel.transform(testing.toDF).show
  }
}
*/

object DecisionTreeExample {

  import org.apache.spark.{SparkConf, SparkContext}
  import org.apache.spark.mllib.util.MLUtils

  def main(args: Array[String]) = {
    val sparkConf: SparkConf = new SparkConf()
    val sc: SparkContext = new SparkContext("local", "TestLocal", sparkConf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    import sqlContext.implicits._

    val fPath = "src" + File.separator + "test" + File.separator + "resources" + File.separator + "sample_libsvm_data.txt"

    val data = sqlContext.createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect).toDF("label", "features")

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val lr = new DecisionTree("decisionTree", sc)
    val lrmodel = lr.fit(trainingData)
    lrmodel.transform(testData).show

  }
}
