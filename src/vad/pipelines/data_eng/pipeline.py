from kedro.pipeline import Pipeline, node
from vad.pipelines.data_eng.nodes import Etl, DataEng


def run_for_train(**kwargs):
    return Pipeline(
        [
            node(
                func=Etl.read_X,
                inputs=[],
                outputs="audio",
                name="read_audio",
            ),
            node(
                func=Etl.load_Y,
                inputs="params:DATA_ENG",
                outputs="label",
                name="load_label",
            ),
            node(
                func=Etl.sync_audio_and_labels,
                inputs=["audio", "label"],
                outputs="synced",
                name="sync_audio_and_label",
            ),
            node(
                func=Etl.test_on_label,
                inputs=["synced", "params:TEST"],
                outputs="tested",
                name="test_label",
            ),
            node(
                func=DataEng.set_resolution,
                inputs=["tested", "params:DATA_ENG"],
                outputs="floated32",
                name="float_and_32bit",
            ),
            node(
                func=DataEng.reshape_input_for_net,
                inputs=["floated32", "params:DATA_ENG"],
                outputs="reshaped",
                name="reshape_for_net",
            ),
            node(
                func=DataEng.split_train_test,
                inputs=["reshaped", "params:DATA_ENG"],
                outputs=[
                    "full_train_audio",
                    "test_audio",
                    "full_train_label",
                    "test_label",
                ],
                name="split-train-test",
            ),
            node(
                func=DataEng.reduce_train,
                inputs=["full_train_audio", "full_train_label", "params:DATA_ENG"],
                outputs=["train_audio", "train_label"],
                name="reduce-train",
            ),
        ]
    )


def run_for_inference(**kwargs):
    return Pipeline(
        [
            node(
                func=Etl.read_X,
                inputs=[],
                outputs="audio",
                name="read_audio",
            ),
            node(
                func=Etl.load_Y,
                inputs="params:DATA_ENG",
                outputs="label",
                name="load_label",
            ),
            node(
                func=Etl.sync_audio_and_labels,
                inputs=["audio", "label"],
                outputs="synced",
                name="sync_audio_and_label",
            ),
            node(
                func=Etl.test_on_label,
                inputs=["synced", "params:TEST"],
                outputs="tested",
                name="test_label",
            ),
            node(
                func=DataEng.set_resolution,
                inputs=["tested", "params:DATA_ENG"],
                outputs="floated32",
                name="float_and_32bit",
            ),
            node(
                func=DataEng.reshape_input_for_net,
                inputs=["floated32", "params:DATA_ENG"],
                outputs="prod_audio",
                name="reshape_for_net",
            ),
        ]
    )
