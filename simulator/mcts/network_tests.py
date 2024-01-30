import unittest
import numpy as np
from network import Network

class TestNetwork(unittest.TestCase):
    """
    Unit test class for the Network class.

    Attributes
    ----------
    network : Network
        The Network instance to be tested.

    Methods
    -------
    setUp():
        Set up the Network instance for testing.
    test_build_model():
        Test the build_model method.
    test_inference():
        Test the inference method.
    test_save_model():
        Test the save_model method.
    test_load_model():
        Test the load_model method.
    test_train():
        Test the train method.
    """

    def setUp(self):
        """
        Set up the Network instance for testing.
        """
        config = [1024, 256, 64, 16]
        self.network = Network(config=config)

    def test_build_model(self):
        """
        Test the build_model method.
        """
        self.network.build_model()
        self.assertIsNotNone(self.network.model, "Model should not be None after build_model")

    def test_inference(self):
        """
        Test the inference method.
        """
        # Assuming the input image is a list of numpy arrays
        images = [np.random.rand(1, 32, 32).flatten() for _ in range(10)]
        value, logits = self.network.inference(images)
        self.assertIsInstance(value, float, "Inference value should be a float")
        self.assertIsInstance(logits, tuple, "Inference logits should be a tuple")
        self.assertEqual(len(logits), 3, "Inference logits should have length 3")

    def test_save_model(self):
        """
        Test the save_model method.
        """
        filepath = "/tmp/test_model.h5"
        result = self.network.save_model(filepath)
        self.assertTrue(result, "Model should be saved successfully")

    def test_load_model(self):
        """
        Test the load_model method.
        """
        filepath = "/tmp/test_model.h5"
        result = self.network.load_model(filepath)
        self.assertTrue(result, "Model should be loaded successfully")

    def test_train(self):
        """
        Test the train method.
        """
        # Create synthetic training data
        images = [np.random.rand(1, 32, 32).flatten() for _ in range(200)]
        labels = [(np.random.rand(), np.random.rand(3)) for _ in range(200)]  # Each label is a tuple
        batch = list(zip(images, labels))

        # Train the model
        self.network.train(batch)

        # Make a prediction with the trained model
        test_image = np.random.rand(1, 32, 32).flatten()
        value, logits = self.network.inference([test_image])

        # Check that the prediction is in the expected format
        self.assertIsInstance(value, float, "Inference value should be a float")
        self.assertIsInstance(logits, tuple, "Inference logits should be a tuple")
        self.assertEqual(len(logits), 3, "Inference logits should have length 3")

if __name__ == "__main__":
    unittest.main()