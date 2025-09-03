from event_system import Pipeline

class Trainer:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Example: dynamically freeze the first layer at index 1
            if epoch > 3:
                print("check")
                self.pipeline.model.set_freeze(layer_names=["fc1"], freeze=True)
            else:
                print("continue")
                self.pipeline.model.set_freeze(layer_names=["fc1"], freeze=False)

            self.pipeline.run_action("train_epoch", epoch=epoch)
            # Register in pipeline
            # SaveModel(filepath=f"checkpoint_epoch{epoch+1}.pth", save_weights_only=True)
            # Call here
            # self.pipeline.run_action("save_model")

    def evaluate(self):
        return self.pipeline.run_action("evaluate")