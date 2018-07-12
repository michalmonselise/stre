import sbt._
import Keys._
import sbt.IvyConsole.Dependencies
import sbtassembly.AssemblyPlugin.autoImport._

scalaVersion := "2.11.8"

//sparkVersion := "2.2.0"

name := "streamingmf"

version := "0.1.0"

licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

resolvers += "Repo at github.com/ankurdave/maven-repo" at "https://github.com/ankurdave/maven-repo/raw/master"

resolvers += "ATG local repository" at "https://nexusrepo.atg-corp.com/repository/maven-public/"

publishTo := Some("Sonatype Nexus Repository Manager" at "https://nexusrepo.atg-corp.com/repository/maven-releases/")

isSnapshot := true

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.4" % "test"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.0" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.2.0" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0" % "provided"

libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"

parallelExecution in Test := false




