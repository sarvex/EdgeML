# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import subprocess

import Common

# Program to build and run the predictor project using msbuild
# The accuracy and other statistics are written to the output file specified

class Predictor:

	def __init__(self, algo, version, datasetType, outputDir):
		self.algo, self.version, self.datasetType = algo, version, datasetType

		self.outputDir = outputDir
		os.makedirs(self.outputDir, exist_ok=True)

	def build(self):
		'''
		Builds using the Predictor.vcxproj project file and creates the executable
		The target platform is currently set to x64
		'''
		print("Build...", end='')

		projFile = "Predictor.vcxproj"
		args = [Common.msbuildPath, projFile, r"/t:Build",
				r"/p:Configuration=Release", r"/p:Platform=x64"]

		logFile = os.path.join(self.outputDir, "msbuild.txt")
		with open(logFile, 'w') as file:
			process = subprocess.call(args, stdout=file)
		
		if process == 1:
			print("FAILED!!\n")
			return False
		else:
			print("success")
			return True

	def execute(self):
		'''
		Invokes the executable with arguments
		'''
		print("Execution...", end='')

		exeFile = os.path.join("x64", "Release", "Predictor.exe")
		args = [exeFile, self.version, self.datasetType]

		logFile = os.path.join(self.outputDir, "exec.txt")
		with open(logFile, 'w') as file:
			process = subprocess.call(args, stdout=file)

		if process == 1:
			print("FAILED!!\n")
			return None
		else:
			print("success")
			acc = self.readStatsFile()
			return acc

	# Read statistics of execution (currently only accuracy)
	def readStatsFile(self):
		statsFile = os.path.join("output", self.version, "stats-" + self.datasetType + ".txt")

		with open(statsFile, 'r') as file:
			content = file.readlines()
			
		stats = [x.strip() for x in content]

		return float(stats[0])

	def run(self):
		res = self.build()
		if res == False:
			return None

		acc = self.execute()

		return acc
