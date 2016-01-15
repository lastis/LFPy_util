var megagen = (function() {
  
  
  
  var lastTime = 0;
  var dt_load = 100;
  var dt_load_tmp = 0;
  var index = 0;
//   var angularSpeed = 0.01;

  var contour_filenames = [];
  
  var loader = new THREE.JSONLoader();
  var scene;
  var camera;
  var renderer;

  var neuron;
  var electrodes;
  var contour;
  var mouse3D;
  var mouse3DText;
  var axisHelper;
  var centerSphere;

  var stats;
  var raycaster;

  var neuronLoaded = false;
  var electrodesLoaded = false;
  var potentialLoaded = false;
  var animationRunning = false;

  var onMouseDownPosition = new THREE.Vector2();
  var isMouseDown = false;

  var offsetX = 0,offsetY=0,offsetZ=0;
  var radius = 500;
  var theta = 45;
  var onMouseDownTheta = 45;
  var phi = 60;
  var onMouseDownPhi = 60;
  

  return {
    
    drawNeuron: function() {
      init();
      loadNeuronMesh();
      loadElectrodesMesh();
      draw()
    }

  }

  function init(){
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 3000 );
    setCameraPosition();

    renderer = new THREE.WebGLRenderer();
    renderer.setSize( window.innerWidth, window.innerHeight );

    raycaster = new THREE.Raycaster();

    var material = new THREE.MeshNormalMaterial();
    material.color = new THREE.Color(0xd3d3d3);
    var sphereGeo = new THREE.SphereGeometry(2);
    mouse3D = new THREE.Mesh(sphereGeo,material);

    centerSphere = new THREE.Mesh(sphereGeo,material);

    document.body.appendChild( renderer.domElement );
    document.addEventListener( 'mousemove', onDocumentMouseMove, false );
    document.addEventListener( 'mousedown', onDocumentMouseDown, false );
    document.addEventListener( 'mouseup', onDocumentMouseUp, false );
    document.addEventListener( 'mousewheel', onDocumentMouseWheel, false );
    document.addEventListener( 'keypress', onDocumentKeyPress, false );
  }

  function loadElectrodesMesh() {
    var json = $.getJSON('electrodes.js', function(data){
      var mergedGeometry = new THREE.Geometry();
      var material = new THREE.MeshNormalMaterial();
      for (i = 0; i < data.x.length; i++) {
        var sphereGeo = new THREE.SphereGeometry(5);
        var sphere = new THREE.Mesh(sphereGeo,material);
        sphere.position.set(data.x[i],data.y[i],data.z[i]);
        sphere.updateMatrix();
        mergedGeometry.merge(sphere.geometry,sphere.matrix);
      }
      electrodes = new THREE.Mesh(mergedGeometry,material);
      electrodesLoaded = true;
      scene.add(electrodes);
      draw();
    });
  }

  function loadNeuronMesh() {
    var json = $.getJSON('neuron.js', function(data){
      var zAxis = new THREE.Vector3(0,0,1);
      var xAxis = new THREE.Vector3(1,0,0);
      var yAxis = new THREE.Vector3(0,1,0);
      var mergedGeometry = new THREE.Geometry();
      var material = new THREE.MeshNormalMaterial();
      // Make the neuron mesh
      var indx = 0;
      for (i = 0; i < data.diam.length-1; i++) {
        var r1 = data.diam[i]/2;
        var r2 = data.diam[i+1]/2;
        if (r1 != 0 && r2 != 0){
          var dx = data.x[indx+1]-data.x[indx];
          var dy = data.y[indx+1]-data.y[indx];
          var dz = data.z[indx+1]-data.z[indx];
          var len = Math.sqrt(
            Math.pow(dx,2) + 
            Math.pow(dy,2) + 
            Math.pow(dz,2)
            ); 
          dx = dx/2;
          dy = dy/2;
          dz = dz/2;
          var x = data.x[indx];
          var y = data.y[indx];
          var z = data.z[indx];
          var x2 = data.x[indx+1];
          var y2 = data.y[indx+1];
          var z2 = data.z[indx+1];
          var xAngle = Math.atan2(Math.sqrt(dx*dx+dz*dz),dy);
          var yAngle = Math.atan2(dx,dz);

          var cylinderGeo = new THREE.CylinderGeometry(r1,r2,len,8,1,false);
          var cylinder = new THREE.Mesh(cylinderGeo,material);
          cylinder.position.set(x+dx,y+dy,z+dz);
          rotateAroundWorldAxis(cylinder,xAxis,xAngle+Math.PI);
          rotateAroundWorldAxis(cylinder,yAxis,yAngle);

          cylinder.updateMatrix();
          mergedGeometry.merge(cylinder.geometry,cylinder.matrix);

//           var sphereGeo = new THREE.SphereGeometry(r1);
//           var sphere = new THREE.Mesh(sphereGeo,material);
//           sphere.position.set(x,y,z);
//           sphere.updateMatrix();
//           mergedGeometry.merge(sphere.geometry,sphere.matrix);
        }
//         if (r1 != 0 && r2 == 0) {
//           var sphereGeo = new THREE.SphereGeometry(r1);
//           var sphere = new THREE.Mesh(sphereGeo,material);
//           sphere.position.set(data[indx].x,data[indx].y,data[indx].z);
//           sphere.updateMatrix();
//           mergedGeometry.merge(sphere.geometry,sphere.matrix);
//         }
        indx++;
      }
      neuron = new THREE.Mesh(mergedGeometry,material);

      axisHelper = new THREE.AxisHelper(100);
      neuronLoaded = true;


      scene.add(neuron);     
      axisHelper.position.x = -50;
      scene.add(axisHelper);
      scene.add(centerSphere);
    });
  }

  function draw(){
    if (animationRunning == false) {
      stats = new Stats();
      stats.setMode(0);
      stats.domElement.style.position = 'absolute';
      stats.domElement.style.left = '0px';
      stats.domElement.style.top = '0px';
      document.body.appendChild(stats.domElement);
      animationRunning = true;
      requestAnimationFrame(animate);
    }
  }

  function animate(){
    stats.begin();
    // update
    var time = (new Date()).getTime();
    var timeDiff = time - lastTime;
    dt_load_tmp = dt_load_tmp + timeDiff;

    lastTime = time;

    renderer.render(scene, camera);
    stats.end();

    requestAnimationFrame(animate);
  }

  function loadContourFromFolder(dir) {
    var filename = contour_filenames[index];
    if (index == contour_filenames.length-1) {
      index = 0;
    }
    loader.load(dir+'/'+filename, function (geo){
      scene.remove(contour);
      contour = new THREE.Mesh(geo, new THREE.MeshNormalMaterial());
      scene.add(contour);
      potentialLoaded = true;
    });
    index++;
  }

  function onDocumentKeyPress(event){
    var key = event.keyCode;
    switch (key){
      case 97:
        offsetX -= 10;
        break
      case 100:
        offsetX += 10;
        break
      case 119:
        offsetY += 10;
        break
      case 115:
        offsetY -= 10;
        break
      case 113:
        offsetZ -= 10;
        break
      case 101:
        offsetZ += 10;
        break
    }
    updateCameraView();
  }

  function onDocumentMouseMove( event ) {
    event.preventDefault();

    if ( isMouseDown ) {
        theta = - ( ( event.clientX - onMouseDownPosition.x ) * 0.5 )
                + onMouseDownTheta;
        phi = ( ( event.clientY - onMouseDownPosition.y ) * 0.5 )
              + onMouseDownPhi;

        updateCameraView();
        camera.updateMatrix();
    }
  }

  function onDocumentMouseDown( event ) {
    event.preventDefault();

    isMouseDown = true;

    onMouseDownTheta = theta;
    onMouseDownPhi = phi;
    onMouseDownPosition.x = event.clientX;
    onMouseDownPosition.y = event.clientY;
    
    updateCameraView();
  }

  function onDocumentMouseUp( event ) {
    event.preventDefault();

    isMouseDown = false;



    onMouseDownPosition.x = event.clientX - onMouseDownPosition.x;
    onMouseDownPosition.y = event.clientY - onMouseDownPosition.y;

    if ( onMouseDownPosition.length() > 5 ) {
        return;
    }

    var mouse = new THREE.Vector2();
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
    raycaster.setFromCamera(mouse,camera);
    var intersects = raycaster.intersectObjects([neuron]);

    if (intersects.length > 0) {
//       mouse3D.position.set(
//           intersects[0].point.x,
//           intersects[0].point.y,
//           intersects[0].point.z
//       );
//       mouse3D.updateMatrix();
//       scene.add(mouse3D);

      var material = new THREE.MeshNormalMaterial();
      var str0 = '(';
      var str1 = intersects[0].point.x.toFixed(2);
      var str2 = ','
      var str3 = intersects[0].point.y.toFixed(2);
      var str4 = ','
      var str5 = intersects[0].point.z.toFixed(2);
      var str6 = ')';
      var str = str0.concat(str1,str2,str3,str4,str5,str6);

      var textGeo = new THREE.TextGeometry(str, {
          size: 10,
          height: 1,
          curveSegments: 4,

          
          font : "optimer",
          weight : "bold", 
          style : "normal",

          bevelThickness: 2,
          bevelSize: 1.5,
          bevelEnabled: true,

          material: 0,
          extrudeMaterial: 1
      });
      textGeo.computeVertexNormals();
      mouse3DText = new THREE.Mesh(textGeo,material);

      mouse3DText.position.set(
          intersects[0].point.x,
          intersects[0].point.y,
          intersects[0].point.z
      );
      scene.add(mouse3DText);

    }
  }

  function onDocumentMouseWheel( event ) {
    radius -= event.wheelDeltaY;
    updateCameraView();
  }

  function updateCameraView(){
    setCameraPosition();
    setCenterCursorPosition();
  }

  function setCenterCursorPosition() {
    centerSphere.position.x = offsetX;
    centerSphere.position.y = offsetY;
    centerSphere.position.z = offsetZ;
  }

  function setCameraPosition() {
    camera.position.x = radius * Math.sin( theta * Math.PI / 360 ) * Math.cos( phi * Math.PI / 360 ) + offsetX;
    camera.position.y = radius * Math.sin( phi * Math.PI / 360 ) + offsetY;
    camera.position.z = radius * Math.cos( theta * Math.PI / 360 ) * Math.cos( phi * Math.PI / 360 ) + offsetZ;
    camera.lookAt(new THREE.Vector3(offsetX,offsetY,offsetZ));
    camera.updateMatrix();
  }

})();

function rotateAroundObjectAxis(object, axis, radians) {
    var rotObjectMatrix = new THREE.Matrix4();
    rotObjectMatrix.makeRotationAxis(axis.normalize(), radians);

    // old code for Three.JS pre r54:
    // object.matrix.multiplySelf(rotObjectMatrix);      // post-multiply
    // new code for Three.JS r55+:
    object.matrix.multiply(rotObjectMatrix);

    // old code for Three.js pre r49:
    // object.rotation.getRotationFromMatrix(object.matrix, object.scale);
    // old code for Three.js r50-r58:
    // object.rotation.setEulerFromRotationMatrix(object.matrix);
    // new code for Three.js r59+:
    object.rotation.setFromRotationMatrix(object.matrix);
}

function rotateAroundWorldAxis(object, axis, radians) {
    var rotWorldMatrix = new THREE.Matrix4();
    rotWorldMatrix.makeRotationAxis(axis.normalize(), radians);

    // old code for Three.JS pre r54:
    //  rotWorldMatrix.multiply(object.matrix);
    // new code for Three.JS r55+:
    rotWorldMatrix.multiply(object.matrix);                // pre-multiply

    object.matrix = rotWorldMatrix;

    // old code for Three.js pre r49:
    // object.rotation.getRotationFromMatrix(object.matrix, object.scale);
    // old code for Three.js pre r59:
    // object.rotation.setEulerFromRotationMatrix(object.matrix);
    // code for r59+:
    object.rotation.setFromRotationMatrix(object.matrix);
}
