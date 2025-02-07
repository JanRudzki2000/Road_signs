#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image;
Rect boundingBox;

// Funkcja do filtrowania obiektów w obrazie binarnym na podstawie ko³owatoœci
void filterCircularObjects(Mat& binaryImage, double minCircularity) {
    // Znalezienie konturów na obrazie binarnym
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Tworzenie nowego obrazu do przechowywania przefiltrowanych obiektów
    Mat filteredImage = Mat::zeros(binaryImage.size(), binaryImage.type());

    // Przechodzenie przez wszystkie znalezione kontury
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);

        // Unikanie dzielenia przez zero
        if (perimeter == 0) continue;

        // Obliczanie wskaŸnika ko³owatoœci
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        // Sprawdzenie, czy obiekt spe³nia kryterium ko³owatoœci
        if (circularity >= minCircularity) {
            // Rysowanie konturu na nowym obrazie
            drawContours(filteredImage, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);
        }
    }

    binaryImage = filteredImage;
}

// Funkcja do filtrowania obiektów w obrazie binarnym na podstawie trójk¹towoœci równobocznej, wiem ¿e inna figura te¿ mo¿e spe³niæ to kryterium, ale na potrzeby tego zadania dok³adnoœæ jest wystarczaj¹ca
Mat filterTriObjects(const Mat& binaryImage, double minTri) {
    // Znalezienie konturów na obrazie binarnym
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Tworzenie nowego obrazu do przechowywania przefiltrowanych obiektów
    Mat filteredImage = Mat::zeros(binaryImage.size(), binaryImage.type());

    // Przechodzenie przez wszystkie znalezione kontury
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        double perimeter = arcLength(contour, true);
        double a = perimeter / 3;

        // Unikanie dzielenia przez zero
        if (perimeter == 0) continue;

        // Obliczanie wskaŸnika trójk¹towoœci równobocznej
        double tri = 12 * sqrt(3) * area / (perimeter * perimeter);
        
        // Sprawdzenie, czy obiekt spe³nia kryterium ko³owatoœci
        if (tri >= minTri && tri <= 2-minTri) {
            // Rysowanie konturu na nowym obrazie
            drawContours(filteredImage, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);
        }
    }

    return filteredImage;
}
// Funkcja do pozostawiania najwiêkszego czerwonego, jest wywo³ywana po znalezieniu niebieskich
Mat keepLargestRed(const Mat& image, Rect boundingBox) {

    Mat imageCrop = image(boundingBox);
    Mat hsv;

    cvtColor(imageCrop, hsv, COLOR_BGR2HSV);

    Scalar lowerRed1 = Scalar(0, 60, 40);
    Scalar upperRed1 = Scalar(3, 255, 255);
    Scalar lowerRed2 = Scalar(170, 60, 40);
    Scalar upperRed2 = Scalar(180, 255, 255);

    Mat mask1, mask2, maskRed;
    //odfiltorwanie czerownych
    inRange(hsv, lowerRed1, upperRed1, mask1);
    inRange(hsv, lowerRed2, upperRed2, mask2);
    //po³¹czenie czerownych z Hue powy¿ej 0 i poni¿ej 180 (360)
    maskRed = mask1 | mask2;
    imshow("czerwone", maskRed);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(maskRed, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    int largestContourIndex = -1;
    double largestFilledArea = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] != -1) {
            continue; // Ignorujemy kontury, które s¹ dziurami (maj¹ rodzica)
        }

        double filledArea = contourArea(contours[i]);

        // Znajdowanie wszystkich dziur zwi¹zanych z tym konturem
        for (size_t j = 0; j < contours.size(); ++j) {
            if (hierarchy[j][3] == (int)i) {
                filledArea -= contourArea(contours[j]);
            }
        }

        if (filledArea > largestFilledArea) {
            largestFilledArea = filledArea;
            largestContourIndex = i;
        }
    }

    vector<Point> largestContour = contours[largestContourIndex];

    Mat largestFragment = Mat::zeros(maskRed.size(), CV_8UC1);
    drawContours(largestFragment, contours, largestContourIndex, Scalar(255), FILLED);

    //imshow("naj czerowny", largestFragment);
    //waitKey(0);
    return largestFragment;
}
// Funckja do pozostawiania najwiêkszego 
Mat keepLargest(const Mat& image) {

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    int largestContourIndex = -1;
    double largestFilledArea = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] != -1) {
            continue; // Ignorujemy kontury, które s¹ dziurami (maj¹ rodzica)
        }

        double filledArea = contourArea(contours[i]);

        // Znajdowanie wszystkich dziur zwi¹zanych z tym konturem
        for (size_t j = 0; j < contours.size(); ++j) {
            if (hierarchy[j][3] == (int)i) {
                filledArea -= contourArea(contours[j]);
            }
        }

        if (filledArea > largestFilledArea) {
            largestFilledArea = filledArea;
            largestContourIndex = i;
        }
    }

    vector<Point> largestContour = contours[largestContourIndex];

    Mat largestFragment = Mat::zeros(image.size(), CV_8UC1);
    drawContours(largestFragment, contours, largestContourIndex, Scalar(255), FILLED);

    return largestFragment;
}
// Funkcja do znajdowania najwiêkszego konturu i jego prostok¹ta otaczaj¹cego
Rect findLargestBoundingBox(const Mat& binaryImage) {
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return Rect(0, 0, 0, 0); // Jeœli brak konturów, zwracamy pusty prostok¹t
    }

    Rect boundingBox = boundingRect(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        boundingBox = boundingBox | boundingRect(contours[i]); // Rozszerzanie prostok¹ta otaczaj¹cego
    }

    return boundingBox;
}

// Funkcja pozostawiaj¹ca tylko najwiêkszy fragment w binarnym obrazie
Mat keepLargest2(const Mat& binaryImage) {
    // Znalezienie wszystkich konturów
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Sprawdzenie, czy znaleziono jakiekolwiek kontury
    if (contours.empty()) {
        return Mat::zeros(binaryImage.size(), CV_8UC1); // Zwrócenie pustego obrazu, jeœli brak konturów
    }

    // Znalezienie konturu o najwiêkszym obszarze
    int largestContourIndex = 0;
    int remis = 0;
    double largestArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        
        if (area > largestArea) {
            largestArea = area;
            largestContourIndex = i;
        }
        
    }
    
    
    Mat largestRed;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        
        double k = (largestArea - area) / largestArea;
        
        if (k < 0.1 && largestContourIndex != i && remis == 0) { //niedpuszczaj¹c fragmentów nieznacznie wiêkszych od obecnie najwiêkszego
            remis = 1;
                        
            boundingBox = findLargestBoundingBox(binaryImage);

            //imshow("po cieciu", binaryImage(boundingBox));
            //waitKey(0);

            largestRed = keepLargestRed(image, boundingBox);
        }

    }
    if(remis ==0) {
       // Utworzenie nowego obrazu binarnego zawieraj¹cego tylko najwiêkszy kontur
       Mat largestFragment = Mat::zeros(binaryImage.size(), CV_8UC1);
       drawContours(largestFragment, contours, largestContourIndex, Scalar(255), FILLED);

       return largestFragment;

    }
    else {
        return largestRed;
    }
}

// Funkcja do znajdowania œrodka masy obiektu w obrazie binarnym
Point2f findCentroid(const Mat& binaryImage) {
    Moments m = moments(binaryImage, true);
    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    return centroid;
}

// Funkcja do znajdowania minimalnego promienia otaczaj¹cego
float findRadius(const Mat& binaryImage, Point2f centroid) {
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    float maxRadius = 0.0;
    for (const auto& contour : contours) {
        Point2f center;
        float radius;
        minEnclosingCircle(contour, center, radius);

        // Sprawdzenie, czy centroid jest wewn¹trz konturu
        if (pointPolygonTest(contour, centroid, false) >= 0) {
            maxRadius = max(maxRadius, radius);
        }
    }

    return maxRadius;
}


int main() {
    // Wczytanie obrazu
    image = imread("D:/!PW/2 semestr/CPO/ostrz1.jpg");
    
    if (image.empty()) {
        std::cerr << "Nie uda³o siê wczytaæ obrazu!" << std::endl;
        return -1;
    }
    //przeskalowanie tak by mieœci³ siê na ekranie komputera
    resize(image, image, { 1200,900 }, 0, 0, INTER_NEAREST);
    //medianBlur(image, image, 15);
    
    Mat imageCopy = image.clone();

    Mat hsv, templateHsv;
    //zmiana przestrzeni barw z BGR do HSV by ³atwiej wyodrêbniæ barwê
    cvtColor(image, hsv, COLOR_BGR2HSV);
    

    int choice;

    while (true) {
        cout << "Napisz 1 jesli wgrales znak zakaz zatrzymywania sie, 2 jesli znak ustap pierszenstwa : ";
        cin >> choice;

        if (choice == 1) { //algorytm znajdowania znaku zakaz zatrzymywania siê
            Scalar lowerRed1 = Scalar(0, 60, 40);
            Scalar upperRed1 = Scalar(3, 255, 255);
            Scalar lowerRed2 = Scalar(170, 60, 40);
            Scalar upperRed2 = Scalar(180, 255, 255);


            Mat mask1, mask2, maskBlack, maskBlue;
            inRange(hsv, lowerRed1, upperRed1, mask1);
            inRange(hsv, lowerRed2, upperRed2, mask2);
            inRange(hsv, Scalar(0, 0, 0), Scalar(255, 255, 20), maskBlack);

            //odfiltrowanie tylko niebieskiego
            inRange(hsv, Scalar(210 / 2, 70, 40), Scalar(230 / 2, 255, 255), maskBlue);

            //Erozja
            int morphSize = 1;
            Mat element = getStructuringElement(MORPH_RECT, Size(2 * morphSize + 1, 2 * morphSize + 1), Point(morphSize, morphSize));
            Mat erod;

            erode(maskBlue, erod, element, Point(-1, -1), 1);

            imshow("tylko niebieski", maskBlue);
            imshow("erozja", erod);
            //imshow("otwarcie", open);

            //Dylatacja
            morphSize = 10; //bylo 5   9
            element = getStructuringElement(MORPH_RECT, Size(2 * morphSize + 1, 2 * morphSize + 1), Point(morphSize, morphSize));

            Mat dill;
            dilate(erod, dill, element, Point(-1, -1), 1);
            imshow("dill", dill);

            // Minimalna ko³owatoœæ do zachowania obiektu
            double minCircularity = 0.7; // Im bli¿ej 1, tym bardziej ko³owaty 0.7 ok

            // Filtrowanie obiektów
            filterCircularObjects(dill, minCircularity);
            imshow("filtrowanie po kszta³cie", dill);

            // Zostawienie tylko najwiêkszego fragmentu
            Mat largestFragment = keepLargest2(dill);
            imshow("tylko najwiekszy", largestFragment);

            // Znajdowanie œrodka masy
            Point2f centroid = findCentroid(largestFragment);

            // Znajdowanie promienia minimalnego okrêgu otaczaj¹cego
            int radius = findRadius(largestFragment, centroid);
            

            Point2f point(centroid.x + boundingBox.x, centroid.y + boundingBox.y);
            vector<vector<Point>> contour;
            vector<Point> largestC;
            findContours(largestFragment, contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            double maxArea = 0;
            for (const auto& cont : contour) {
                double area = cv::contourArea(cont);
                if (area > maxArea) {
                    maxArea = area;
                    largestC = cont;
                }
            }
            // Rysowanie œrodka masy na obrazie
            Mat colorImage;
            RotatedRect elli;
            elli = fitEllipse(largestC);
            elli.center.x += boundingBox.x; //przesuwanie œrodka elipsy tak by poprawnie wygl¹da³a na oryginalnym obrazie
            elli.center.y += boundingBox.y;
            // konwersja do BGR by móc wyrysowaæ kolorow¹ elipsê i okr¹g
            cvtColor(largestFragment, colorImage, COLOR_GRAY2BGR);

            circle(image, point, 5, Scalar(0, 255, 0), -1); // œrodek masy
            //circle(image, point, radius, Scalar(0, 255, 0), 2); // okr¹g (kontur)
            ellipse(image, elli, Scalar(0, 255, 0), 2);

            int pointX = point.x;
            int pointY = point.y;
            int centroidX = centroid.x;
            int centroidY = centroid.y;

            imshow("zaznaczone", image);
            waitKey(0);

            cout << "Wspolrzedne znaku (x: " << pointX << " y:" << pointY << " )" << endl;

            break;
        }
        else if (choice == 2) {//algorytm znajdowania znaku ust¹p pierwszeñstwa
            Scalar lowerY = Scalar(30 / 2, 60, 40);
            Scalar upperY = Scalar(60 / 2, 255, 255);

            Mat mask;

            //odfiltrowanie tylko ¿ó³tego
            inRange(hsv, lowerY, upperY, mask);

            imshow("tylko zó³ty", mask);


            // Otwarcie
            int morphSize = 1; //lub 2
            Mat element = getStructuringElement(MORPH_RECT, Size(2 * morphSize + 1, 2 * morphSize + 1), Point(morphSize, morphSize));
            Mat open, filtr, open2;

            morphologyEx(mask, open, MORPH_OPEN, element, Point(-1, -1), 2);

            imshow("otwarcie", open);

            filtr = filterTriObjects(open, 0.8); //im bli¿ej 1 tym bli¿ej trójk¹ta równoramiennego, ograniczenie z do³u
            imshow("fitr trójk¹tów", filtr);

            // Otwarcie nr 2
            morphSize = 4;
            element = getStructuringElement(MORPH_RECT, Size(2 * morphSize + 1, 2 * morphSize + 1), Point(morphSize, morphSize));

            morphologyEx(filtr, open2, MORPH_OPEN, element, Point(-1, -1), 2);

            imshow("otwarcie 2", open2);

            Rect boundingBox2;// boundingBox bez 2 jest zmienn¹ globaln¹
            boundingBox2 = findLargestBoundingBox(open2);

            Mat imageCrop = open(boundingBox2);// celowo open(1) a nie filtr, ¿eby nie by³y zalany

            Mat largest;
            largest = keepLargest(imageCrop);
            imshow("crop", imageCrop);
            imshow("najwiêkszy", largest);

            vector<vector<Point>> contours;

            findContours(largest, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (auto& kontur : contours) {
                for (auto& punkt : kontur) {
                    punkt.x += boundingBox2.x; //przesuwanie konturu by poprawnie wygl¹da³ na oryginalnym obrazie
                    punkt.y += boundingBox2.y;
                }
            }

            Point2f centroid = findCentroid(largest);

            Point2f point(centroid.x + boundingBox2.x, centroid.y + boundingBox2.y);
            // Rysowanie œrodka masy na obrazie
            Mat colorImage;

            circle(image, point, 5, Scalar(0, 255, 0), -1); // œrodek masy

            int pointX = point.x;
            int pointY = point.y;
            int centroidX = centroid.x;
            int centroidY = centroid.y;

            
            cout << "Wspolrzedne znaku (x: " << pointX << " y:" << pointY << " )" << endl;
            // Rysowanie konturów na kolorowym obrazie
            for (size_t i = 0; i < contours.size(); ++i) {
                drawContours(image, contours, static_cast<int>(i), Scalar(0, 255, 0), 3);
            }

            imshow("zaznaczone", image);
            waitKey(0);

            break;
        }
        else {
            cout << "Wprowadzona wartoœæ jest nieprawid³owa" << endl;
        }
    }




    return 0;
}