for K in 100 500 1300 2500 5000; do
    build/Release/sample_image/sample_image sample_image/girl_with_a_pearl.jpg -n ${K} -o captures/voronoi/girl_with_a_pearl_${K}.jpg
    build/Release/sample_image/sample_image sample_image/lady.jpg              -n ${K} -o captures/voronoi/lady_${K}.jpg             
    build/Release/sample_image/sample_image sample_image/mona_lisa.jpg         -n ${K} -o captures/voronoi/mona_lisa_${K}.jpg        
    build/Release/sample_image/sample_image sample_image/starry_night.jpg      -n ${K} -o captures/voronoi/starry_night_${K}.jpg     
    build/Release/sample_image/sample_image sample_image/girl_with_a_pearl.jpg -n ${K} -d -o captures/voronoi/girl_with_a_pearl_${K}_improved.jpg
    build/Release/sample_image/sample_image sample_image/lady.jpg              -n ${K} -d -o captures/voronoi/lady_${K}_improved.jpg             
    build/Release/sample_image/sample_image sample_image/mona_lisa.jpg         -n ${K} -d -o captures/voronoi/mona_lisa_${K}_improved.jpg        
    build/Release/sample_image/sample_image sample_image/starry_night.jpg      -n ${K} -d -o captures/voronoi/starry_night_${K}_improved.jpg     
    build/Release/sample_image/sample_image sample_image/girl_with_a_pearl.jpg -n ${K} -p -o captures/voronoi/girl_with_a_pearl_${K}_points.jpg
    build/Release/sample_image/sample_image sample_image/lady.jpg              -n ${K} -p -o captures/voronoi/lady_${K}_points.jpg             
    build/Release/sample_image/sample_image sample_image/mona_lisa.jpg         -n ${K} -p -o captures/voronoi/mona_lisa_${K}_points.jpg        
    build/Release/sample_image/sample_image sample_image/starry_night.jpg      -n ${K} -p -o captures/voronoi/starry_night_${K}_points.jpg     
    build/Release/sample_image/sample_image sample_image/girl_with_a_pearl.jpg -n ${K} -p -d -o captures/voronoi/girl_with_a_pearl_${K}_improved_points.jpg
    build/Release/sample_image/sample_image sample_image/lady.jpg              -n ${K} -p -d -o captures/voronoi/lady_${K}_improved_points.jpg             
    build/Release/sample_image/sample_image sample_image/mona_lisa.jpg         -n ${K} -p -d -o captures/voronoi/mona_lisa_${K}_improved_points.jpg        
    build/Release/sample_image/sample_image sample_image/starry_night.jpg      -n ${K} -p -d -o captures/voronoi/starry_night_${K}_improved_points.jpg     
done