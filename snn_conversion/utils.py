def evaluate_conversion(converted_model, x_test, y_test, testacc, batch_size, timesteps=50):
    for i in range(1, timesteps + 1):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
            "acc spiking (orig): %.2f%% (%.2f%%)" % (acc * 100, testacc * 100),
            "- conv loss: %+.2f%%" % ((-(1 - acc / testacc) * 100))
        )


def evaluate_conversion_and_save_data(converted_model, x_test, y_test, testacc, batch_size, timesteps=50):

    """
    Utility function for evaluation and saving of the simulation accuracy.
    """

    accuracy_per_t = []
    for i in range(0, timesteps):
        _, acc = converted_model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        accuracy_per_t.append(acc)
        print(
            "Timesteps", str(i) + "/" + str(timesteps) + " -",
            "acc spiking (orig): %.2f%% (%.2f%%)" % (acc*100, testacc*100),
            "- conv loss: %+.2f%%" % ((-(1 - acc/testacc)*100)))
    return accuracy_per_t