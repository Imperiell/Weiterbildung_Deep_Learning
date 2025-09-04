from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from actions import *
from module_wrapper import InteractiveModule
from event_system import *
from trainer import *

if __name__ == "__main__":
    # Dummy-Daten!!!
    X_train = torch.randn(500, 20)
    y_train = torch.randint(0, 2, (500,))
    X_test = torch.randn(100, 20)
    y_test = torch.randint(0, 2, (100,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    seq_model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        ),
        nn.Linear(64, 2)
    )
    model = InteractiveModule(seq_model)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Pipeline & Actions
    pipeline = Pipeline(model, optimizer, train_loader, test_loader, criterion, device="cpu")
    pipeline.register_action("train_epoch", TrainEpoch())
    pipeline.register_action("evaluate", Evaluate())

    # Events
    pipeline.register_event(FreezingLogger())
    # pipeline.register_event(ModelCheckpoint(directory="checkpoints"))
    pipeline.register_event(MetricsLogger())

    # Trainer
    trainer = Trainer(pipeline)
    trainer.train(num_epochs=5)
    trainer.evaluate()