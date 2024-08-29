from flask import Flask, jsonify
import os

from . import main
from .training_load import TrainingLoad

load = TrainingLoad(0.8)

@main.route('/initialize', methods=["GET", "POST"])
def initialize():
    # Init
    if not load.hasSavedModel:
        load.initModel(8)
        trainX, _ = load.read_in_csv()
        load.fitWithData(trainX)
    else:
        load.loadModel()


@main.route('/training_load', methods=["POST"])
def get_training_load():
    info = load.export()
    return jsonify(info), 204