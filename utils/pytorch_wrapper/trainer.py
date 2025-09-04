from event_system import Pipeline

class Trainer:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
    #TODO: evtl generalisieren! training per name -> meh!
    #TODO: evtl funktion auslagern, die dann in main geschrieben wird
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            # Example: dynamically freeze layer
            if epoch > 3:
                print("check")
                self.pipeline.model.set_freeze(layer_names=["Linear_0"], freeze=True)
            else:
                print("continue")
                self.pipeline.model.set_freeze(layer_names=["Linear_0"], freeze=False)

            self.pipeline.run_action("train_epoch", epoch=epoch)
            # Register in pipeline
            # SaveModel(filepath=f"checkpoint_epoch{epoch+1}.pth", save_weights_only=True)
            # Call here
            # self.pipeline.run_action("save_model")

    def evaluate(self):
        return self.pipeline.run_action("evaluate")