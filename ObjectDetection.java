/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.examples.inference;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.math.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;

/**
 * An example of inference using an object detection model.
 *
 * <p>See this <a
 * href="https://github.com/awslabs/djl/blob/master/examples/docs/object_detection.md">doc</a> for
 * information about this example.
 */
public final class ObjectDetection {

    private static final Logger logger = LoggerFactory.getLogger(ObjectDetection.class);

    private ObjectDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = ObjectDetection.predict();
        logger.info("{}", detection);
    }


    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        //Path imageFile = Paths.get("src/test/resources/pikachu.jpg");
        //Image img = ImageFactory.getInstance().fromFile(imageFile);

        static SingleShotDetectionTranslator.Builder builder = SingleShotDetectionTranslator.builder()
                .addTransform(new ToTensor())
                .optSynset(Collections.singletonList("pikachu"))
                .optThreshold(0.7f);

        static SingleShotDetectionTranslator translator = new SingleShotDetectionTranslator(builder){

            @Override
            public NDList processInput(TranslatorContext ctx, Image input){
                NDList list = super.processInput(ctx, input);
                NDArray array = list.get(0).expandDims(0);
                return new NDList(array);
            }

            @Override
            public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
                NDArray anchors = list.get(0);
                NDArray classPredictions = list.get(1).softmax(-1).transpose(0, 2, 1);
                NDArray boundingBoxPredictions = list.get(2);
                MultiBoxDetection multiBoxDetection =
                        MultiBoxDetection.builder().build();
                NDList detections =
                        multiBoxDetection.detection(
                                new NDList(
                                        classPredictions,
                                        boundingBoxPredictions,
                                        anchors));
                list = detections.singletonOrThrow().split(new long[]{1, 2},2);
                NDList output = new NDList(list.size());
                for (NDArray array : list) {
                    output.add(array.squeeze(0));
                }
                return super.processOutput(ctx, list);
            }
            @Override
            public Batchifier getBatchifier(){
                return null;
            }
        };

        Block block = getSsdTrainBlock();
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optBlock(block)
                        .optModelPath(Paths.get("build/model/"))
                        .optModelName("pikachu-ssd")
                        .optTranslator(translator)
                        .build();

        try(ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
            Predictor<Image, DetectedObjects> predictor = model.newPredictor(translator)) {
            Image image = ImageFactory.getInstance().fromUrl("src/test/resources/pikachu.jpg");
            DetectedObjects detectedObjects = predictor.predict(image);
            image.drawBoundingBoxes(detectedObjects);
            Path out = Paths.get("/build/output").resolve("detected-pikachu.png");
            image.save(Files.newOutputStream(out), "png");
            return detectedObjects.getNumberOfObjects();

        }
    }

    private static Block getSsdTrainBlock() {
        int[] numFilters = {16, 32, 64};
        SequentialBlock baseBlock = new SequentialBlock();
        for (int numFilter : numFilters) {
            baseBlock.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }

        List<List<Float>> sizes = new ArrayList<>();
        List<List<Float>> ratios = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ratios.add(Arrays.asList(1f, 2f, 0.5f));
        }
        sizes.add(Arrays.asList(0.2f, 0.272f));
        sizes.add(Arrays.asList(0.37f, 0.447f));
        sizes.add(Arrays.asList(0.54f, 0.619f));
        sizes.add(Arrays.asList(0.71f, 0.79f));
        sizes.add(Arrays.asList(0.88f, 0.961f));

        return SingleShotDetection.builder()
                .setNumClasses(1)
                .setNumFeatures(3)
                .optGlobalPool(true)
                .setRatios(ratios)
                .setSizes(sizes)
                .setBaseNetwork(baseBlock)
                .build();
    }

    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {
        Path outputDir = Paths.get("build/output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected-pikachu.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        logger.info("Detected objects image has been saved in: {}", imagePath);
    }
}
