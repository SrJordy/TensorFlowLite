package com.example.tensorflowlite;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.tensorflowlite.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_EXTERNAL_STORAGE_PERMISSION = 101;
    private static final int REQUEST_CAMERA_CAPTURE = 1;
    private static final int REQUEST_GALLERY_CAPTURE = 2;

    private TextView result, confidence;
    private ImageView imageView;
    private Button picture, buttonGallery;
    private static final int IMAGE_SIZE = 224;
    private static final String[] CLASSES = {"Pera", "Manzana", "Kiwi", "Fresa", "Banana"};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        setupButtonListeners();
    }

    private void initializeViews() {
        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        buttonGallery = findViewById(R.id.button_gallery);
    }

    private void setupButtonListeners() {
        picture.setOnClickListener(view -> requestCameraPermissionAndOpenCamera());
        buttonGallery.setOnClickListener(view -> requestStoragePermissionAndOpenGallery());
    }

    private void requestCameraPermissionAndOpenCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    private void requestStoragePermissionAndOpenGallery() {
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            openGallery();
        } else {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_EXTERNAL_STORAGE_PERMISSION);
        }
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA_CAPTURE);
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, REQUEST_GALLERY_CAPTURE);
    }

    private void classifyImage(Bitmap image) {
        try {
            ByteBuffer byteBuffer = getByteBufferFromImage(image);
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, IMAGE_SIZE, IMAGE_SIZE, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            displayClassificationResults(outputFeature0.getFloatArray());
            model.close();
        } catch (IOException e) {
            throw new RuntimeException("Error in initializing the model", e);
        }
    }

    private ByteBuffer getByteBufferFromImage(Bitmap image) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        int pixel = 0;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }

    private void displayClassificationResults(float[] confidences) {
        int maxPos = 0;
        float maxConfidence = 0;
        for (int i = 0; i < confidences.length; i++) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i];
                maxPos = i;
            }
        }
        result.setText(CLASSES[maxPos]);

        StringBuilder s = new StringBuilder();
        for (int i = 0; i < CLASSES.length; i++) {
            s.append(String.format("%s: %.1f%%\n", CLASSES[i], confidences[i] * 100));
        }
        confidence.setText(s.toString());
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Bitmap image = null;
            if (requestCode == REQUEST_CAMERA_CAPTURE) {
                image = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == REQUEST_GALLERY_CAPTURE) {
                image = getImageFromData(data);
            }

            if (image != null) {
                displayImageAndClassify(image);
            }
        }
    }

    private Bitmap getImageFromData(Intent data) {
        try {
            return MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());
        } catch (IOException e) {
            throw new RuntimeException("Error in getting image from gallery", e);
        }
    }

    private void displayImageAndClassify(Bitmap image) {
        int dimension = Math.min(image.getWidth(), image.getHeight());
        image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
        imageView.setImageBitmap(image);

        image = Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false);
        classifyImage(image);
    }
}
